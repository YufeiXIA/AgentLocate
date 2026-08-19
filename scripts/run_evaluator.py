#!/usr/bin/env python3
"""Run the configurable Who&When evaluator ensemble and write a text report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentlocate.data import load_traces
from agentlocate.evaluator import PROMPT_TYPES, evaluate_votes, load_predictions, source_item
from agentlocate.judge import make_backend
from run_initial_judge import resolve_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="Initial Judge .txt output.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "evaluator",
                        help="Directory for generated Evaluator text reports.")
    parser.add_argument("--evaluator-output", "--laj-log", "--output", dest="evaluator_output", type=Path,
                        default=None, help="Optional explicit .txt Evaluator report path.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("auto", "transformers", "openai"), default="auto")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--n-votes", "--n-evaluators", dest="n_votes", type=int, default=3)
    parser.add_argument("--agent-prompts", nargs="+", choices=PROMPT_TYPES,
                        default=["base", "concise", "evidence_first"])
    parser.add_argument("--step-prompts", nargs="+", choices=PROMPT_TYPES, default=["cot"])
    parser.add_argument("--vote-strategy", choices=("hybrid", "majority", "confidence"), default="hybrid")
    parser.add_argument("--aggregation-method", choices=("method1", "method2"), default="method1")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", "--max_tokens", dest="max_new_tokens", type=int, default=768)
    parser.add_argument("--device-map", "--device_map", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quantize", choices=("none", "8bit", "4bit"), default="none")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--offload-dir", type=Path, default=None)
    parser.add_argument("--max-gpu-mem", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--limit-files", type=int, default=10)
    args = parser.parse_args()
    if args.n_votes < 1:
        parser.error("--n-votes must be at least 1")
    traces = load_traces(args.dataset_dir, args.split_file)
    if args.fast:
        traces = traces[:args.limit_files]
    predictions = load_predictions(args.predictions)
    backend_name, model = resolve_backend(args.backend, args.model)
    backend = make_backend(backend_name, model, args.temperature, args.max_new_tokens, args.device_map,
                           args.top_p, args.device, args.quantize, args.cpu_offload,
                           str(args.offload_dir) if args.offload_dir else None, args.max_gpu_mem, args.api_key)
    items = []
    for trace in traces:
        prediction = predictions.get(trace.case_id, {"agent": "Unknown", "step": None})
        votes = evaluate_votes(trace, backend, str(prediction.get("agent") or "Unknown"), prediction.get("step"),
                               args.n_votes, args.agent_prompts, args.step_prompts)
        items.append(source_item(trace, prediction, votes, args.vote_strategy, args.aggregation_method))
    total = max(1, len(items))
    summary = {"total": len(items), "hits": sum(bool(item.get("judge_correct_on_gt")) for item in items),
               "hit_rate": sum(bool(item.get("judge_correct_on_gt")) for item in items) / total,
               "vote_strategy": args.vote_strategy, "aggregation_method": args.aggregation_method,
               "votes_per_case": args.n_votes, "temperature": args.temperature,
               "agent_prompt_types": args.agent_prompts, "step_prompt_types": args.step_prompts,
               "agent_accuracy": sum(bool(item.get("agent_match")) for item in items) / total,
               "step_accuracy": sum(bool(item.get("step_match")) for item in items) / total}
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model.replace("/", "-")).strip("-")
    initial_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.predictions.stem).strip("-")
    output = args.evaluator_output or args.output_dir / f"{model_slug}_under_{initial_slug}.txt"
    if output.suffix.lower() != ".txt":
        parser.error("Evaluator output must use the .txt format.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "top_improvement_tips": [], "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} evaluated cases to {output}")
    print("Convert this report to SFT JSONL with scripts/convert_evaluator_txt_to_sft.py.")


if __name__ == "__main__":
    main()
