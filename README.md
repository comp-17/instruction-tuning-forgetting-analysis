# Sequential Instruction Tuning of a Small LLM with Forgetting Analysis

**Student:** Susheela Sri Akunuru (fpb170)  
**Instructor:** Dr. Peyman Najafirad (Paul Rad)  

---

## Overview

This repository implements a two-stage sequential instruction-tuning pipeline for Phi-3.5 Mini Instruct, investigating catastrophic forgetting across three training checkpoints. A teacher-generated JSON dataset is constructed through imitation learning from Llama 3.3 70B Instruct (UTSA-hosted). Evaluation is performed using both automatic metrics (ROUGE, BERTScore) and a strong LLM judge following the Self-Instruct evaluation protocol (Taori et al., 2023).

**Core Research Question:**
> Does fine-tuning a small LLM on Alpaca data followed by teacher-generated JSON data improve structured-output reliability while maintaining general instruction-following ability, or does catastrophic forgetting occur?

**Key Finding:** Stage 2 JSON fine-tuning achieves near-perfect structured output validity scores (4.98/5 judge score) with no catastrophic forgetting — ROUGE metrics actually improved from Checkpoint 1 to Checkpoint 2 (+4.5% ROUGE-L), and BERTScore dropped by only 0.7%. Sequential fine-tuning on a small, high-quality teacher-generated dataset proved additive rather than destructive.

---

## Pipeline Architecture

```
Base Phi-3.5 Mini Instruct (Checkpoint 0 — baseline)
              │
              ▼
Stage 1: QLoRA on Alpaca-Cleaned
         46,584 examples, 2 epochs, lr=2e-5
              │
              ▼
Checkpoint 1: Alpaca-tuned model
              │
              ▼
Stage 2: QLoRA on Teacher-Generated JSON Data
         112 examples × 10 epochs, lr=2e-5
         Teacher: Llama 3.3 70B Instruct (imitation learning)
              │
              ▼
Checkpoint 2: JSON-tuned model
```

---

## Repository Structure

```
instruction-tuning-forgetting-analysis/
├── config.yaml                  # All hyperparameters in one place
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── REPORT.md                    # GitHub blog post (full experimental report)
│
├── data/
│   ├── alpaca_train.json        # Stage 1 training data (46,584 examples)
│   ├── alpaca_eval.json         # Alpaca held-out evaluation set
│   ├── json_train.json          # Stage 2 training data (112 examples)
│   ├── json_eval.json           # JSON held-out evaluation set (13 examples)
│   └── json_eval_full.json      # Full JSON evaluation set (125 examples)
│
├── prompts/
│   ├── json_extraction.txt      # Teacher prompt: JSON extraction
│   ├── schema_generation.txt    # Teacher prompt: Schema-constrained generation
│   ├── classification.txt       # Teacher prompt: Classification with JSON output
│   ├── json_repair.txt          # Teacher prompt: JSON repair
│   ├── tool_call.txt            # Teacher prompt: Tool-call argument generation
│   ├── judge_alpaca.txt         # Judge prompt: General instruction evaluation
│   └── judge_json.txt           # Judge prompt: Structured JSON evaluation
│
├── scripts/
│   ├── slurm_stage1.sh          # UTSA HPC SLURM batch script — Stage 1
│   └── slurm_stage2.sh          # UTSA HPC SLURM batch script — Stage 2
│
├── figures/
│   ├── stage1_train_loss.png    # W&B Stage 1 training loss curve
│   ├── stage1_eval_loss.png     # W&B Stage 1 evaluation loss curve
│   ├── stage2_train_loss.png    # W&B Stage 2 training loss curve
│   ├── stage2_eval_loss.png     # W&B Stage 2 evaluation loss curve
│   ├── rouge_l_comparison.png   # ROUGE-L across three checkpoints (auto-generated)
│   ├── bertscore_comparison.png # BERTScore across three checkpoints (auto-generated)
│   ├── json_validity_comparison.png # JSON validity across checkpoints (auto-generated)
│   └── forgetting_analysis.png  # C1 vs C2 forgetting analysis (auto-generated)
│
├── prepare_alpaca.py            # Stage 1 data preparation
├── generate_teacher_data.py     # Imitation learning pipeline (Stage 2 data)
├── train_stage1.py              # Stage 1 QLoRA training
├── train_stage2.py              # Stage 2 QLoRA training
├── train_combined.py            # Combined training ablation (joint Alpaca + JSON)
├── inference.py                 # Response generation at each checkpoint
├── judge_eval.py                # LLM-as-a-Judge pairwise evaluation
├── compute_metrics.py           # Automatic metrics computation
├── ablation.py                  # Ablation study runner (epochs)
├── generate_figures.py          # Auto-generates all result figures from metrics
│
└── results/
    ├── checkpoint0_alpaca.json  # Checkpoint 0 outputs on Alpaca eval
    ├── checkpoint0_json.json    # Checkpoint 0 outputs on JSON eval
    ├── checkpoint1_alpaca.json  # Checkpoint 1 outputs on Alpaca eval
    ├── checkpoint1_json.json    # Checkpoint 1 outputs on JSON eval
    ├── checkpoint2_alpaca.json  # Checkpoint 2 outputs on Alpaca eval
    ├── checkpoint2_json.json    # Checkpoint 2 outputs on JSON eval
    ├── judge_*.json             # Judge evaluation results
    ├── all_metrics.json         # All computed automatic metrics
    └── ablations/               # Ablation study results
```

---

## Setup Instructions

### Prerequisites

- UTSA ARC HPC access (request at `hpcsupport.utsa.edu`)
- GPU node: Tesla V100S-PCIE-32GB recommended
- UTSA VPN: required for Llama 3.3 70B API access
- Python 3.10

### Step 1: Connect to UTSA ARC

```bash
ssh fpb170@arc.utsa.edu
```

After authentication (UTSA passphrase + Duo), the session lands in `/work/fpb170/`.

### Step 2: Request a GPU Node

Navigate to `portal.arc.utsa.edu` → My Interactive Sessions → Arc Virtual Desktop. Select partition `gpu1v100`, set duration to 72 hours, and launch. Note the assigned node name (e.g., `gpu015`).

### Step 3: Set Up the Conda Environment

```bash
module load anaconda3
conda create -p /work/fpb170/.conda/envs/llm python=3.10 -y
conda activate /work/fpb170/.conda/envs/llm
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

The following variables must be set in `~/.bashrc` on ARC:

```bash
export CONDA_ENVS_DIRS=/work/fpb170/.conda/envs
export HF_HOME=/work/fpb170/hf_cache
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
export WANDB_PROJECT=assignment3

# UTSA-hosted models (no VPN required)
export UTSA_API_KEY_8B=your_8b_api_key
export UTSA_BASE_URL_8B=http://149.165.173.247:8888/v1
export UTSA_MODEL_8B=meta-llama/Llama-3.1-8B-Instruct

# UTSA-hosted models (VPN required)
export UTSA_API_KEY_70B=your_70b_api_key
export UTSA_BASE_URL_70B=http://10.246.100.230/v1
export UTSA_MODEL_70B=llama-3.3-70b-instruct-awq
```

---

## Reproduction Steps

All results reported in `REPORT.md` can be reproduced by executing the following commands in order:

```bash
cd /work/fpb170/assignment3

# 1. Prepare Stage 1 Alpaca data
python prepare_alpaca.py

# 2. Generate Stage 2 teacher JSON data (UTSA VPN required)
python generate_teacher_data.py

# 3. Generate Checkpoint 0 baseline outputs
python inference.py --checkpoint checkpoint0

# 4. Train Stage 1 (Alpaca fine-tuning)
python train_stage1.py

# 5. Generate Checkpoint 1 outputs
python inference.py --checkpoint checkpoint1

# 6. Train Stage 2 (Teacher JSON fine-tuning)
python train_stage2.py

# 7. Generate Checkpoint 2 outputs
python inference.py --checkpoint checkpoint2

# 8. Run judge evaluation (UTSA VPN required)
python judge_eval.py --comparison all

# 9. Compute all automatic metrics
python compute_metrics.py

# 10. Run ablation study (epochs)
python ablation.py --ablation epochs

# 11. Run combined training ablation
python train_combined.py

# 12. Run inference on combined checkpoint
python inference.py --checkpoint combined

# 13. Generate all result figures
python generate_figures.py
```

### Running via SLURM (Batch Jobs)

```bash
# Submit Stage 1 training as a batch job
sbatch scripts/slurm_stage1.sh

# Monitor job status
squeue -u fpb170

# Submit Stage 2 after Stage 1 completes
sbatch scripts/slurm_stage2.sh
```

---

## Model and Training Configuration

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Student model | microsoft/Phi-3.5-mini-instruct | — (continued from Stage 1) |
| Teacher model | — | Llama-3.3-70B-Instruct-awq |
| Judge model | — | Llama-3.3-70B-Instruct-awq |
| Fine-tuning method | QLoRA (4-bit NF4) | QLoRA (4-bit NF4) |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.05 | 0.05 |
| Trainable parameters | 25,165,824 (0.65%) | 25,165,824 (0.65%) |
| Epochs | 2 | 10 |
| Learning rate | 2e-5 | 2e-5 |
| Batch size | 4 | 4 |
| Gradient accumulation | 4 | 4 |
| Max sequence length | 1024 | 1024 |
| LR scheduler | cosine | cosine |
| Total steps | 5,822 | 70 |
| Training time | ~4 hours | ~6 minutes |

---

## Dataset Statistics

| Dataset | Split | Examples |
|---|---|---|
| Alpaca-Cleaned | Train | 46,584 |
| Alpaca-Cleaned | Eval (used) | 100 |
| Teacher JSON | Train | 112 |
| Teacher JSON | Eval (full) | 125 |
| Teacher JSON | Total | 125 |

**Teacher JSON task breakdown (25 examples per task type):**
- JSON Extraction from unstructured text
- Schema-Constrained Generation
- Exact-Label Classification with JSON output
- JSON Repair and Formatting Correction
- Tool-Call Argument Generation

---

## Key Experimental Results

### Three-Checkpoint Comparison

| Checkpoint | ROUGE-L | BERTScore F1 | JSON Validity | Judge Win Rate |
|---|---|---|---|---|
| Checkpoint 0 (base) | 0.1596 | 0.8506 | 97.6% | 80.0% (vs C1) |
| Checkpoint 1 (Alpaca) | 0.1072 | 0.8384 | 96.0% | 11.0% (vs C0) |
| Checkpoint 2 (Sequential JSON) | **0.1120** | 0.8322 | **98.4%** | 49.0% (vs C1) |
| Combined (Joint training) | 0.1083 | 0.8370 | 97.6% | 30.0% (vs C2) |

### Forgetting Analysis (C1 → C2)

| Metric | Change | Verdict |
|---|---|---|
| ROUGE-L | +4.5% | MAINTAINED |
| BERTScore F1 | −0.7% | MAINTAINED |
| Judge Win Rate | C2 wins 49% vs C1's 25% | IMPROVED |

### Combined Training Ablation (Sequential vs Joint)

| Comparison | Sequential C2 Wins | Combined Wins | Verdict |
|---|---|---|---|
| Alpaca evaluation | **42.0%** | 30.0% | Sequential wins |
| JSON evaluation | **35.2%** | 18.4% | Sequential wins |

**Key finding:** Sequential two-stage training consistently outperforms joint combined training, validating the two-stage pipeline design.

### JSON Judge Scores (Checkpoint 2)

| Dimension | Score |
|---|---|
| Instruction Following | 4.74 / 5 |
| Correctness | **4.78 / 5** |
| Clarity | 4.81 / 5 |
| Completeness | 4.72 / 5 |
| Structured Output Validity | **4.98 / 5** |
| Hallucination Risk | 4.89 / 5 |

---

## UTSA HPC Training Details

| Detail | Value |
|---|---|
| Cluster | UTSA ARC |
| GPU | Tesla V100S-PCIE-32GB |
| CUDA | 12.3 |
| Stage 1 training time | ~4 hours |
| Stage 2 training time | ~6 minutes |
| Combined training time | ~4 hours |
| Job scheduler | SLURM |

---

## References

1. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*
2. Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv:2305.14314*
3. Taori et al. (2023). Alpaca: A Strong, Replicable Instruction-Following Model. *Stanford CRFM*
4. Wang et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions. *ACL 2023*
5. Gu et al. (2024). A Survey on LLM-as-a-Judge. *arXiv:2411.15594*
