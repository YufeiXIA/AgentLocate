#!/usr/bin/env python3
"""Convert an Evaluator text report into the source-compatible SFT JSONL format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentlocate.data import load_traces
from agentlocate.evaluator import export_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-output", type=Path, required=True, help="Evaluator .txt report.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "evaluator" / "sft")
    parser.add_argument("--sft-output", type=Path, default=None, help="Optional explicit JSONL path.")
    parser.add_argument("--prefer-judge-label", action="store_true",
                        help="Use evaluator corrections instead of the benchmark ground truth labels.")
    args = parser.parse_args()
    if args.evaluator_output.suffix.lower() != ".txt":
        parser.error("--evaluator-output must be a .txt report.")
    report = json.loads(args.evaluator_output.read_text(encoding="utf-8"))
    items = report.get("items")
    if not isinstance(items, list):
        parser.error("Evaluator report has no items list.")
    traces = load_traces(args.dataset_dir, args.split_file)
    output = args.sft_output or args.output_dir / f"{args.evaluator_output.stem}.jsonl"
    export_sft(items, {trace.case_id: trace for trace in traces}, output, prefer_gt=not args.prefer_judge_label)
    print(f"Wrote {len(items)} source-compatible SFT records to {output}")


if __name__ == "__main__":
    main()
