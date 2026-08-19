#!/usr/bin/env python3
"""Run the Initial Judge on a fixed Who&When split."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentlocate.data import load_traces
from agentlocate.judge import make_backend, run_initial_judge, write_initial_text

MODEL_ALIASES = {
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}


def resolve_backend(backend: str, model: str) -> tuple[str, str]:
    model = MODEL_ALIASES.get(model, model)
    if backend == "auto":
        backend = "openai" if model.startswith(("gpt-", "ft:gpt-", "o1", "o3")) else "transformers"
    return backend, model


def output_name(model: str, method: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model.replace("/", "-"))
    return f"{slug.strip('-')}_{method}.txt"


def build_parser(*, description: str = __doc__, default_output_dir: Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-dir", "--directory-path", "--directory_path", type=Path, required=True,
                        help="Directory containing original Who&When JSON traces.")
    parser.add_argument("--split-file", type=Path, required=True,
                        help="Fixed train/test index file to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir or ROOT / "output" / "initial_judge",
                        help="Directory for generated Initial Judge text outputs.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional explicit .txt output path; normally use --output-dir.")
    parser.add_argument("--method", choices=("all_at_once", "step_by_step"), required=True)
    parser.add_argument("--model", required=True,
                        help="Original alias (qwen-7b, llama-8b, mistral-7b, ...) or model identifier.")
    parser.add_argument("--backend", choices=("auto", "transformers", "openai"), default="auto")
    parser.add_argument("--api-key", "--api_key", default=None,
                        help="OpenAI API key; defaults to OPENAI_API_KEY. Never stored in this repository.")
    parser.add_argument("--max-new-tokens", "--max_tokens", dest="max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device-map", "--device_map", default="auto")
    parser.add_argument("--device", default=None,
                        help="Local device when --device-map none (e.g. cuda:0 or cpu).")
    parser.add_argument("--quantize", choices=("none", "8bit", "4bit"), default="none")
    parser.add_argument("--cpu-offload", "--cpu_offload", action="store_true")
    parser.add_argument("--offload-dir", "--offload_dir", type=Path, default=None)
    parser.add_argument("--max-gpu-mem", "--max_gpu_mem", default=None,
                        help="Optional per-GPU memory limit for Accelerate, e.g. 38GiB.")
    parser.add_argument("--adapter-path", "--adapter_path", type=Path, default=None,
                        help="Optional LoRA adapter directory produced by train_judge_lora.py.")
    parser.add_argument("--fast", action="store_true",
                        help="Use only --limit-files traces, matching the source quick-check mode.")
    parser.add_argument("--limit-files", "--limit_files", type=int, default=10,
                        help="Number of traces used with --fast (default: 10).")
    return parser


def run(args: argparse.Namespace) -> None:

    traces = load_traces(args.dataset_dir, args.split_file)
    if args.fast:
        traces = traces[: args.limit_files]
    backend_name, model = resolve_backend(args.backend, args.model)
    backend = make_backend(
        backend=backend_name,
        model=model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        top_p=args.top_p,
        device=args.device,
        quantize=args.quantize,
        cpu_offload=args.cpu_offload,
        offload_dir=str(args.offload_dir) if args.offload_dir else None,
        max_gpu_mem=args.max_gpu_mem,
        api_key=args.api_key,
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
    )
    records = run_initial_judge(traces, backend, args.method, backend_name, model)
    output = args.output or args.output_dir / output_name(args.model, args.method)
    if output.suffix.lower() != ".txt":
        parser.error("Initial Judge output must use the .txt format.")
    write_initial_text(records, output)
    print(f"Wrote {len(records)} predictions to {output}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
