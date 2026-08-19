"""Initial Judge for all-at-once and step-by-step failure attribution."""

from __future__ import annotations

import abc
import json
import os
import re
from functools import cache
from pathlib import Path
from typing import Any

from .data import Trace, format_history


class JudgeBackend(abc.ABC):
    """Minimal completion interface shared by local and API-backed models."""

    @abc.abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant completion for an OpenAI-style message list."""


class OpenAIJudge(JudgeBackend):
    """Initial Judge backed by an OpenAI chat-completions model."""

    def __init__(
        self, model: str, temperature: float = 0.6, max_tokens: int = 1024,
        top_p: float = 0.95, api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Install openai to use backend=openai.") from exc
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for backend=openai.")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
        )
        return response.choices[0].message.content or ""


class TransformersJudge(JudgeBackend):
    """Initial Judge for local Hugging Face models such as Qwen or Llama."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.6,
        max_new_tokens: int = 1024,
        top_p: float = 0.95,
        device_map: str = "auto",
        device: str | None = None,
        quantize: str = "none",
        cpu_offload: bool = False,
        offload_dir: str | None = None,
        max_gpu_mem: str | None = None,
        adapter_path: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Install torch and transformers to use backend=transformers.") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        options: dict[str, Any] = {"torch_dtype": "auto", "trust_remote_code": True}
        if device_map != "none":
            options["device_map"] = device_map
        if max_gpu_mem and torch.cuda.is_available() and device_map == "auto":
            options["max_memory"] = {0: max_gpu_mem}
        if cpu_offload:
            if not offload_dir:
                raise ValueError("--cpu-offload requires --offload-dir.")
            options["offload_folder"] = offload_dir
        if quantize != "none":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("Install bitsandbytes for --quantize=8bit or 4bit.") from exc
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=quantize == "8bit",
                load_in_4bit=quantize == "4bit",
                llm_int8_enable_fp32_cpu_offload=cpu_offload,
            )
        self._model = AutoModelForCausalLM.from_pretrained(model, **options)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("Install peft to load a LoRA adapter.") from exc
            adapter_tokenizer_config = Path(adapter_path) / "tokenizer_config.json"
            if adapter_tokenizer_config.is_file():
                # LoRA training may add a PAD token or otherwise resize the
                # vocabulary.  Load the checkpoint tokenizer and resize the
                # base embeddings before PEFT restores adapter-side embeddings.
                self._tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
                if self._tokenizer.pad_token_id is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                if self._model.get_input_embeddings().weight.shape[0] != len(self._tokenizer):
                    self._model.resize_token_embeddings(len(self._tokenizer))
            self._model = PeftModel.from_pretrained(self._model, adapter_path)
        if device_map == "none" and device:
            self._model.to(device)
        self._model.eval()
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._top_p = top_p

    def complete(self, messages: list[dict[str, str]]) -> str:
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = self._temperature > 0
        with self._torch.inference_mode():
            options: dict[str, Any] = {
                "do_sample": do_sample,
                "max_new_tokens": self._max_new_tokens,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if do_sample:
                options.update(temperature=self._temperature, top_p=self._top_p)
            generated = self._model.generate(**inputs, **options)
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


def make_backend(
    backend: str,
    model: str,
    temperature: float,
    max_new_tokens: int,
    device_map: str,
    top_p: float = 0.95,
    device: str | None = None,
    quantize: str = "none",
    cpu_offload: bool = False,
    offload_dir: str | None = None,
    max_gpu_mem: str | None = None,
    api_key: str | None = None,
    adapter_path: str | None = None,
) -> JudgeBackend:
    if backend == "openai":
        return OpenAIJudge(model, temperature, max_new_tokens, top_p, api_key)
    if backend == "transformers":
        return TransformersJudge(
            model, temperature, max_new_tokens, top_p, device_map, device, quantize,
            cpu_offload, offload_dir, max_gpu_mem, adapter_path,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _json_object(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _step(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "initial_judge"


@cache
def _template(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def _render(name: str, **values: object) -> str:
    prompt = _template(name)
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", str(value))
    return prompt


def full_messages(trace: Trace) -> list[dict[str, str]]:
    return [{"role": "user", "content": _render(
        "all_at_once.txt", problem=trace.question, failure_log=format_history(trace)
    )}]


def step_messages(trace: Trace, step: int) -> list[dict[str, str]]:
    turn = trace.turns[step]
    return [{"role": "user", "content": _render(
        "step_by_step.txt",
        problem=trace.question,
        conversation_history=format_history(trace, stop_at=step),
        step_index=step,
        agent_name=turn.agent,
    )}]


def _labelled(raw: str, label: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", raw)
    return match.group(1).strip() if match else ""


def _known_agent(candidate: str, trace: Trace) -> str:
    for turn in trace.turns:
        if candidate.casefold() == turn.agent.casefold():
            return turn.agent
    return "Unknown"


def judge_all_at_once(trace: Trace, backend: JudgeBackend) -> tuple[dict[str, Any], str]:
    raw = backend.complete(full_messages(trace))
    parsed = _json_object(raw)
    agent = str(parsed.get("agent") or parsed.get("agent_name") or _labelled(raw, "Agent Name")).strip()
    step_value = parsed["step"] if "step" in parsed else _labelled(raw, "Step Number")
    step = _step(step_value)
    rationale = str(parsed.get("rationale") or _labelled(raw, "Reason for Mistake"))
    valid_steps = {turn.index for turn in trace.turns}
    agent = _known_agent(agent, trace)
    if step not in valid_steps:
        step = None
    return {"agent": agent, "step": step, "rationale": rationale}, raw


def judge_step_by_step(trace: Trace, backend: JudgeBackend) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    for turn in trace.turns:
        raw = backend.complete(step_messages(trace, turn.index))
        parsed = _json_object(raw)
        verdict = re.search(r"(?im)^\s*1\.\s*(yes|no)\b", raw)
        has_error = parsed.get("has_error") is True or (verdict is not None and verdict.group(1).lower() == "yes")
        rationale = str(parsed.get("rationale") or _labelled(raw, "2. Reason"))
        audit.append({"step": turn.index, "agent": turn.agent, "has_error": has_error,
                      "rationale": rationale, "raw": raw})
        if has_error:
            return {"agent": turn.agent, "step": turn.index, "rationale": rationale}, audit
    return {"agent": "Unknown", "step": None, "rationale": "No decisive error selected."}, audit


def run_initial_judge(
    traces: list[Trace],
    backend: JudgeBackend,
    method: str,
    backend_name: str,
    model: str,
) -> list[dict[str, Any]]:
    """Run one initial Judge protocol and return JSONL-ready records."""

    records: list[dict[str, Any]] = []
    for trace in traces:
        if method == "all_at_once":
            prediction, raw = judge_all_at_once(trace, backend)
            audit: list[dict[str, Any]] | None = None
        elif method == "step_by_step":
            prediction, audit = judge_step_by_step(trace, backend)
            raw = None
        else:
            raise ValueError(f"Unsupported method: {method}")
        records.append(
            {
                "case_id": trace.case_id,
                "question": trace.question,
                "method": method,
                "backend": backend_name,
                "model": model,
                "prediction": prediction,
                "raw_response": raw,
                "step_audit": audit,
            }
        )
    return records


def write_jsonl(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_initial_text(records: list[dict[str, Any]], output: Path) -> None:
    """Write the source-compatible Initial Judge text format used by the evaluator."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            prediction = record["prediction"]
            handle.write(f"Prediction for {record['case_id']}.json:\n")
            handle.write(f"Agent Name: {prediction.get('agent', 'Unknown')}\n")
            handle.write(f"Step Number: {prediction.get('step') if prediction.get('step') is not None else 'None'}\n")
            handle.write(f"Reason for Mistake: {prediction.get('rationale', '')}\n")
            if record.get("raw_response") is not None:
                handle.write(f"Raw Response: {record['raw_response']}\n")
            handle.write("\n")
