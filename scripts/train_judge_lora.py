#!/usr/bin/env python3
"""Fine-tune a Who&When Judge adapter from evaluator-derived SFT JSONL."""

from __future__ import annotations

import os

if os.environ.get("AGENTLOCATE_REQUIRE_SLURM") == "1" and not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Run GPU training through a Slurm allocation, not on the Zurada login node.")

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.replace("/", "-")).strip("-")


def _config_defaults(argv: list[str] | None) -> dict[str, Any]:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    known, _ = probe.parse_known_args(argv)
    if not known.config:
        return {}
    if not known.config.is_file():
        raise FileNotFoundError(f"Training config not found: {known.config}")
    value = yaml.safe_load(known.config.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Training config must contain a YAML mapping.")
    return value


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional YAML defaults; CLI values take precedence.")
    parser.add_argument("--base-model", "--base_model", required=True, help="Local path or Hugging Face model ID.")
    parser.add_argument("--train-file", "--train_file", type=Path, required=True, help="Evaluator-derived SFT JSONL.")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None, help="LoRA adapter directory.")
    parser.add_argument("--epochs", "--num-train-epochs", "--num_train_epochs", type=float, default=1)
    parser.add_argument("--batch-size", "--batch_size", "--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", "--grad_accum", "--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--cutoff-len", "--cutoff_len", "--max-seq-length", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--no-shuffle-fraction", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", "--lora_r", type=int, default=16)
    parser.add_argument("--lora-alpha", "--lora_alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", "--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", "--lora_target_modules",
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj")
    parser.add_argument("--quantize", choices=("none", "4bit", "8bit"), default="4bit")
    parser.add_argument("--load-in-4bit", "--load_in_4bit", action="store_true")
    parser.add_argument("--load-in-8bit", "--load_in_8bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                        help="Use FP16; enabled automatically on CUDA unless --bf16 is set.")
    parser.add_argument("--device-map", "--device_map", default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-strategy", choices=("steps", "epoch", "no"), default="steps")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs; use 1 for a smoke test.")
    parser.add_argument("--resume", choices=("auto", "never", "path"), default="auto")
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    parser.set_defaults(**defaults)
    return parser


def format_example(tokenizer: AutoTokenizer, sample: dict[str, Any]) -> str:
    if isinstance(sample.get("messages"), list):
        messages = sample["messages"]
    else:
        instruction = str(sample.get("instruction") or "").strip()
        extra_input = str(sample.get("input") or "").strip()
        output = sample.get("output")
        if not instruction or output is None:
            raise ValueError("Each SFT record needs messages or instruction/output fields.")
        messages = [
            {"role": "system", "content": "You are a helpful assistant for failure attribution."},
            {"role": "user", "content": instruction if not extra_input else f"{instruction}\n\n{extra_input}"},
            {"role": "assistant", "content": str(output)},
        ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def load_dataset_for_training(tokenizer: AutoTokenizer, path: Path, args: argparse.Namespace):
    if not path.is_file():
        raise FileNotFoundError(f"SFT file not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"SFT file is empty: {path}")
    dataset = Dataset.from_dict({"text": [format_example(tokenizer, record) for record in records]})
    if not 0 < args.train_fraction <= 1:
        raise ValueError("--train-fraction must be in (0, 1].")
    if args.train_fraction < 1:
        if not args.no_shuffle_fraction:
            dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.select(range(max(1, int(len(dataset) * args.train_fraction))))
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1).")
    split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed) if args.val_ratio else {"train": dataset}

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.cutoff_len, padding=False)

    train = split["train"].map(tokenize, batched=True, remove_columns=["text"])
    valid = split.get("test")
    if valid is not None:
        valid = valid.map(tokenize, batched=True, remove_columns=["text"])
    return train, valid, len(records)


def load_lora_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True,
                                              local_files_only=args.local_files_only)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    quantize = "4bit" if args.load_in_4bit else "8bit" if args.load_in_8bit else args.quantize
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Only one of --load-in-4bit and --load-in-8bit may be set.")
    quantization = None
    if quantize == "4bit":
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    elif quantize == "8bit":
        quantization = BitsAndBytesConfig(load_in_8bit=True)
    options: dict[str, Any] = {"trust_remote_code": True, "local_files_only": args.local_files_only,
                               "low_cpu_mem_usage": True}
    if args.device_map != "none":
        options["device_map"] = args.device_map
    if quantization is not None:
        options["quantization_config"] = quantization
    else:
        options["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **options)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    if quantization is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=modules, bias="none", task_type="CAUSAL_LM"))
    return model, tokenizer


def main(argv: list[str] | None = None) -> None:
    args = build_parser(_config_defaults(argv)).parse_args(argv)
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16.")
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    output_dir = args.output_dir or ROOT / "output" / "fine_tuned_judge" / f"{_slug(args.base_model)}-lora"
    model, tokenizer = load_lora_model(args)
    train_data, valid_data, original_records = load_dataset_for_training(tokenizer, args.train_file, args)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    fp16 = torch.cuda.is_available() and not args.bf16 if args.fp16 is None else args.fp16
    training = TrainingArguments(
        output_dir=str(output_dir), per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, num_train_epochs=args.epochs, max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps, logging_first_step=True,
        save_strategy=args.save_strategy, save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        eval_strategy="steps" if valid_data is not None else "no",
        eval_steps=args.eval_steps if valid_data is not None else None,
        bf16=args.bf16, fp16=fp16, tf32=torch.cuda.is_available(), report_to="none",
        gradient_checkpointing=args.gradient_checkpointing, optim="paged_adamw_8bit" if args.quantize != "none" else "adamw_torch",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training, train_dataset=train_data, eval_dataset=valid_data,
                      data_collator=collator)
    checkpoint = None
    if args.resume == "auto" and output_dir.is_dir():
        checkpoint = get_last_checkpoint(str(output_dir))
    elif args.resume == "path":
        if not args.resume_path or not args.resume_path.is_dir():
            raise ValueError("--resume path requires an existing --resume-path directory.")
        checkpoint = str(args.resume_path)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    summary = {"base_model": args.base_model, "train_file": str(args.train_file), "records": original_records,
               "train_examples": len(train_data), "eval_examples": len(valid_data) if valid_data is not None else 0,
               "seed": args.seed, "quantize": args.quantize, "lora_r": args.lora_r,
               "lora_alpha": args.lora_alpha, "max_steps": args.max_steps}
    (Path(output_dir) / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter and training summary to {output_dir}")


if __name__ == "__main__":
    main()
