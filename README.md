# AgentLocate

Official code for the COLM 2026 submission *Who Broke the System? Failure Localization in LLM-Based Multi-Agent Systems*.

The repository contains the Who&When benchmark, fixed splits, prompts, and the Initial Judge → Evaluator → SFT → LoRA → PEFT Judge pipeline.

## Setup

```powershell
conda env create -f environment.yml
conda activate agentlocate
```

Use your own model path or API credentials. Do not commit model weights, API keys, or generated outputs.

## Run

Run from the project root. Replace `<backend>` and `<model>` with your own configuration. For an API model, set `OPENAI_API_KEY` in your shell, use `--backend openai`, and provide the API model name through `--model`.

```powershell
# 1. Initial Judge
python .\scripts\run_initial_judge.py `
  --dataset-dir .\benchmark\Algorithm-Generated `
  --split-file .\benchmark\splits\alg_test_indices_seed1500.txt `
  --output .\output\initial_judge\initial_all_at_once.txt `
  --method all_at_once --backend <backend> --model "<model>" --quantize 4bit --temperature 0

# 2. Evaluator
python .\scripts\run_evaluator.py `
  --dataset-dir .\benchmark\Algorithm-Generated `
  --split-file .\benchmark\splits\alg_test_indices_seed1500.txt `
  --predictions .\output\initial_judge\initial_all_at_once.txt `
  --evaluator-output .\output\evaluator\evaluator.txt `
  --backend <backend> --model "<model>" --quantize 4bit `
  --n-votes 3 --agent-prompts base concise evidence_first --step-prompts cot `
  --vote-strategy hybrid --aggregation-method method1 --temperature 0

# 3. Evaluator report to SFT data
python .\scripts\convert_evaluator_txt_to_sft.py `
  --evaluator-output .\output\evaluator\evaluator.txt `
  --dataset-dir .\benchmark\Algorithm-Generated `
  --split-file .\benchmark\splits\alg_test_indices_seed1500.txt `
  --output-dir .\output\evaluator\sft

# 4. LoRA fine-tuning
python .\scripts\train_judge_lora.py `
  --config .\configs\lora_train.example.yaml `
  --base-model "<model>" `
  --train-file .\output\evaluator\sft\evaluator.jsonl `
  --output-dir .\output\fine_tuned_judge\adapter --quantize 4bit

# 5. Final PEFT Judge
python .\scripts\run_fine_tuned_judge.py `
  --dataset-dir .\benchmark\Algorithm-Generated `
  --split-file .\benchmark\splits\alg_test_indices_seed1500.txt `
  --method all_at_once --backend transformers --model "<model>" `
  --adapter-path .\output\fine_tuned_judge\adapter `
  --output-dir .\output\fine_tuned_judge --quantize 4bit --temperature 0
```

## Citation

```bibtex
@inproceedings{xia2026agentlocate,
  title     = {Who Broke the System? Failure Localization in {LLM}-Based Multi-Agent Systems},
  author    = {Xia, Yufei and Gao, Anjun and Quan, Yueyang and Liu, Zhuqing and Fang, Minghong},
  booktitle = {Conference on Language Modeling ({COLM})},
  year      = {2026},
  note      = {Submission}
}
```
