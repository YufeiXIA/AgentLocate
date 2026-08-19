"""Who&When evaluator ensemble and source-compatible SFT export."""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

from .data import Trace
from .judge import JudgeBackend


PROMPT_TYPES = ("base", "concise", "evidence_first", "cot")
_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "evaluator"


def format_evaluator_log(trace: Trace) -> str:
    """Use one-based line labels but retain the benchmark's zero-based steps."""
    return "\n".join(
        f"L{turn.index + 1:03d} [Step {turn.index}, {turn.agent}]: {turn.content}"
        for turn in trace.turns
    )


def compact_log_by_hints(trace: Trace, hint_lines: list[int], *, predicted_agent: str | None = None,
                         window: int = 2, head: int = 6, tail: int = 6, cap: int = 80) -> str:
    """Reproduce the source SFT context selection from evaluator evidence lines.

    The source implementation keeps a three-line radius around each cited
    ``rationale_points[].line`` despite exposing ``window=2`` in its signature.
    We intentionally retain that behavior so the exported instruction text
    matches the reference pipeline.
    """
    indexed: list[tuple[int, str, str]] = []
    for one_based_line, turn in enumerate(trace.turns, start=1):
        content = re.sub(r"\s+", " ", turn.content or "").strip()
        if content:
            indexed.append((one_based_line, turn.agent, content))
    if not indexed:
        return "(no history available)"

    keep: set[int] = set()
    line_count = len(indexed)
    for hint_line in hint_lines:
        for line in range(max(1, hint_line - 3), min(line_count, hint_line + 3) + 1):
            keep.add(line)
    if predicted_agent:
        keep.update(line for line, agent, _ in indexed if agent == predicted_agent)
    if not keep:
        keep.update(range(1, min(head, line_count) + 1))
        keep.update(range(max(1, line_count - tail + 1), line_count + 1))

    contents = {line: (agent, content) for line, agent, content in indexed}
    return "\n".join(
        f"L{line:03d} [{contents[line][0]}]: {contents[line][1]}"
        for line in sorted(keep)[:cap]
        if line in contents
    )


@cache
def _prompt_template(prompt_type: str) -> tuple[str, str]:
    text = (_PROMPT_DIR / f"{prompt_type}.txt").read_text(encoding="utf-8")
    system, marker, user = text.partition("\n[USER]\n")
    if not marker or not system.startswith("[SYSTEM]\n"):
        raise ValueError(f"Invalid evaluator prompt template: {prompt_type}")
    return system.removeprefix("[SYSTEM]\n").strip(), user.strip()


def build_evaluator_messages(trace: Trace, predicted_agent: str, prompt_type: str) -> list[dict[str, str]]:
    """Build the source prompt families with one common, parseable output schema."""
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"Unsupported prompt type {prompt_type!r}; choose from {PROMPT_TYPES}.")
    system, user = _prompt_template(prompt_type)
    values = {
        "task": trace.question,
        "failure_log": format_evaluator_log(trace),
        "predicted_agent": predicted_agent,
        "max_step": len(trace.turns) - 1,
    }
    for key, value in values.items():
        user = user.replace("{{" + key + "}}", str(value))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _extract_object(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _as_step(value: Any, trace: Trace, fallback: int | None) -> int | None:
    try:
        step = int(value)
    except (TypeError, ValueError):
        return fallback
    return step if 0 <= step < len(trace.turns) else fallback


def _known_agent(value: Any, trace: Trace, fallback: str) -> str:
    candidate = str(value or "").strip()
    for turn in trace.turns:
        if candidate.casefold() == turn.agent.casefold():
            return turn.agent
    return fallback


def normalize_verdict(raw: str, trace: Trace, predicted_agent: str, predicted_step: int | None) -> dict[str, Any]:
    """Apply the same symbolic guardrails as the source evaluator."""
    value = _extract_object(raw)
    agent = _known_agent(value.get("correct_agent"), trace, predicted_agent)
    step = _as_step(value.get("correct_step"), trace, predicted_step)
    points = value.get("rationale_points")
    if not isinstance(points, list) or not points:
        points = [{"line": 1, "why": "No structured evidence was returned by the evaluator."}]
    tips = value.get("improvement_tips")
    if not isinstance(tips, list) or not tips:
        tips = ["Review the identified failure step and add a validation check."]
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "is_correct": bool(value.get("is_correct", False)),
        "correct_agent": agent,
        "correct_step": step,
        "confidence": min(1.0, max(0.0, confidence)),
        "rationale": str(value.get("rationale", "")),
        "rationale_points": points,
        "improvement_tips": [str(tip) for tip in tips],
    }


def evaluate_votes(
    trace: Trace,
    backend: JudgeBackend,
    predicted_agent: str,
    predicted_step: int | None,
    n_votes: int,
    agent_prompts: list[str],
    step_prompts: list[str],
) -> list[dict[str, Any]]:
    """One source-style vote = one Agent evaluator plus one Step evaluator."""
    votes: list[dict[str, Any]] = []
    for vote_id in range(n_votes):
        agent_type = agent_prompts[vote_id % len(agent_prompts)]
        step_type = step_prompts[vote_id % len(step_prompts)]
        agent = normalize_verdict(backend.complete(build_evaluator_messages(trace, predicted_agent, agent_type)),
                                  trace, predicted_agent, predicted_step)
        step = normalize_verdict(backend.complete(build_evaluator_messages(trace, predicted_agent, step_type)),
                                 trace, predicted_agent, predicted_step)
        votes.append({
            "correct_agent": agent["correct_agent"],
            "correct_step": step["correct_step"],
            "confidence": (agent["confidence"] + step["confidence"]) / 2,
            "is_correct": agent["is_correct"],
            "agent_rationale": agent["rationale"],
            "step_rationale": step["rationale"],
            "rationale_points": agent["rationale_points"],
            "step_rationale_points": step["rationale_points"],
            "improvement_tips": agent["improvement_tips"],
            "_agent_method": agent_type,
            "_step_method": step_type,
            "_vote_id": vote_id,
        })
    return votes


def aggregate_votes(votes: list[dict[str, Any]], predicted_agent: str, predicted_step: int | None,
                    strategy: str, aggregation_method: str) -> tuple[bool, str, int | None, dict[str, Any]]:
    """Method 1 follows the source majority/confidence hybrid rule; Method 2 votes direct pairs."""
    if not votes:
        return False, predicted_agent, predicted_step, {"reason": "no_votes"}
    if aggregation_method == "method2":
        pairs = Counter((v["correct_agent"], v["correct_step"]) for v in votes)
        best_count = max(pairs.values())
        candidates = [pair for pair, count in pairs.items() if count == best_count]
        agent, step = max(candidates, key=lambda pair: sum(v["confidence"] for v in votes if (v["correct_agent"], v["correct_step"]) == pair))
        return agent == predicted_agent and step == predicted_step, agent, step, {
            "strategy": "method2_direct", "pairs": {f"{a}@{s}": count for (a, s), count in pairs.items()}
        }
    true_votes = [v for v in votes if v["is_correct"]]
    false_votes = [v for v in votes if not v["is_correct"]]
    counts = {"true": len(true_votes), "false": len(false_votes)}
    weights = {key: (sum(v["confidence"] for v in group) / len(group) if group else 0.0)
               for key, group in (("true", true_votes), ("false", false_votes))}
    accept = counts["true"] > counts["false"]
    if counts["true"] == counts["false"]:
        accept = weights["true"] >= weights["false"] if strategy != "majority" else True
    if strategy == "confidence":
        accept = weights["true"] >= weights["false"]
    if accept:
        return True, predicted_agent, predicted_step, {"counts": counts, "weights": weights, "strategy": f"{strategy}_accept"}
    pairs = Counter((v["correct_agent"], v["correct_step"]) for v in false_votes)
    best_count = max(pairs.values())
    candidates = [pair for pair, count in pairs.items() if count == best_count]
    agent, step = max(candidates, key=lambda pair: sum(v["confidence"] for v in false_votes if (v["correct_agent"], v["correct_step"]) == pair))
    return False, agent, step, {
        "counts": counts, "weights": weights, "strategy": f"{strategy}_reject",
        "pairs": {f"{a}@{s}": count for (a, s), count in pairs.items()},
    }


def source_item(trace: Trace, prediction: dict[str, Any], votes: list[dict[str, Any]], strategy: str,
                aggregation_method: str) -> dict[str, Any]:
    pred_agent = str(prediction.get("agent") or "Unknown")
    pred_step = _as_step(prediction.get("step"), trace, None)
    correct, agent, step, debug = aggregate_votes(votes, pred_agent, pred_step, strategy, aggregation_method)
    agent_points = [point for vote in votes for point in vote["rationale_points"]]
    step_points = [point for vote in votes for point in vote["step_rationale_points"]]
    tips = list(dict.fromkeys(tip for vote in votes for tip in vote["improvement_tips"]))
    confidence = sum(vote["confidence"] for vote in votes) / len(votes)
    item: dict[str, Any] = {
        "file": f"{trace.case_id}.json", "task": trace.question,
        "log_text": "\n".join(f"{turn.index + 1}: {turn.content}" for turn in trace.turns),
        "mistake_reason": "", "available_steps_zero_based": list(range(len(trace.turns))),
        "predicted_agent": pred_agent, "predicted_step": pred_step,
        "judge_is_correct_agent": correct, "judge_correct_agent": agent,
        "judge_confidence": confidence, "judge_rationale_points": agent_points,
        "judge_step": step, "judge_step_confidence": confidence,
        "judge_step_rationales": step_points, "judge_tips": tips,
        "is_correct": correct, "correct_agent": agent, "correct_step": step,
        "confidence": confidence, "votes": votes,
        "vote_strategy": strategy, "aggregation_method": aggregation_method,
        "aggregation_debug": debug,
    }
    if trace.mistake_agent is not None:
        item.update({"gt_agent": trace.mistake_agent, "gt_step": trace.mistake_step,
                     "agent_match": agent.casefold() == trace.mistake_agent.casefold(),
                     "step_match": step == trace.mistake_step,
                     "judge_correct_on_gt": (correct and pred_agent.casefold() == trace.mistake_agent.casefold()) or
                                            (not correct and agent.casefold() == trace.mistake_agent.casefold())})
    return item


def export_sft(items: list[dict[str, Any]], traces: dict[str, Trace], output: Path, prefer_gt: bool = True) -> None:
    """Export the identical instruction/output record schema used by run_hybrid_method1_pipeline.py."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            trace = traces[item["file"].removesuffix(".json")]
            hint_lines: list[int] = []
            for vote in item.get("votes", []):
                for point in vote.get("rationale_points", []):
                    value = point.get("line") if isinstance(point, dict) else None
                    if isinstance(value, int) or isinstance(value, str) and value.isdigit():
                        hint_lines.append(int(value))
            compact_log = compact_log_by_hints(trace, hint_lines, predicted_agent=item["predicted_agent"])
            use_gt = prefer_gt and item.get("gt_agent") is not None
            agent = str(item["gt_agent"] if use_gt else item["correct_agent"])
            step = str(item["gt_step"] if use_gt else item["correct_step"])
            source = "gt" if use_gt else "judge"
            instruction = f"""Task: {trace.question}

Conversation Log (compacted):
{compact_log}

Initial Prediction:
- Agent: {item['predicted_agent']}
- Step: {item['predicted_step']}

Please analyze whether this prediction is correct, and if not, provide the correct attribution."""
            record = {"instruction": instruction, "output": f"Agent: {agent}\nStep: {step}\nSource: {source}",
                      "task": trace.question, "predicted_agent": item["predicted_agent"],
                      "predicted_step": item["predicted_step"], "label_agent": agent, "label_step": step,
                      "label_source": source, "confidence": item["confidence"],
                      "is_correct": item["is_correct"], "case_id": trace.case_id}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".txt":
        records: dict[str, dict[str, Any]] = {}
        content = path.read_text(encoding="utf-8")
        for block in re.split(r"(?=^Prediction for )", content, flags=re.MULTILINE):
            case = re.search(r"^Prediction for\s+(.+?)\.json:\s*$", block, flags=re.MULTILINE)
            if not case:
                continue
            agent = re.search(r"^Agent Name:\s*(.*?)\s*$", block, flags=re.MULTILINE)
            step = re.search(r"^Step Number:\s*(.*?)\s*$", block, flags=re.MULTILINE)
            records[case.group(1)] = {
                "agent": agent.group(1) if agent else "Unknown",
                "step": step.group(1) if step else None,
            }
        return records
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        prediction = value.get("prediction", value)
        case_id = str(value.get("case_id") or value.get("id") or "")
        if case_id:
            records[case_id] = {"agent": prediction.get("agent") or prediction.get("agent_name"),
                                "step": prediction.get("step")}
    return records
