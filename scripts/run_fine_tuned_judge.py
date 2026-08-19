#!/usr/bin/env python3
"""Run a final Who&When Judge from a base model and its PEFT LoRA adapter."""

from __future__ import annotations

import re
from pathlib import Path

from run_initial_judge import ROOT, build_parser, run


def _adapter_output_name(adapter_path: Path, method: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", adapter_path.name).strip("-")
    return f"{slug or 'fine_tuned_judge'}_{method}.txt"


def main() -> None:
    parser = build_parser(
        description=__doc__,
        default_output_dir=ROOT / "output" / "fine_tuned_judge",
    )
    adapter_action = next(action for action in parser._actions if action.dest == "adapter_path")
    adapter_action.required = True
    args = parser.parse_args()
    if args.backend == "openai":
        parser.error("A PEFT adapter requires --backend transformers (or --backend auto with a local model).")
    if args.output is None:
        args.output = args.output_dir / _adapter_output_name(args.adapter_path, args.method)
    run(args)


if __name__ == "__main__":
    main()
