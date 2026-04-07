# Instruction Tuning & Forgetting Analysis

**Course:** LLM & Agentic Systems — Graduate Course, UTSA  
**Student:** fpb170  
**Instructor:** Dr. Peyman Najafirad (Paul Rad)  
**TA:** Mohammad Bahrami  

---

## Overview

This repository implements a **two-stage sequential instruction-tuning pipeline** for a small language model (Phi-3.5 Mini Instruct), investigating catastrophic forgetting across three checkpoints using both automatic metrics and a strong LLM judge (Llama 3.3 70B).

**Core Research Question:**
> Does fine-tuning a small LLM on Alpaca data followed by teacher-generated JSON data improve structured-output reliability while maintaining general instruction-following ability, or does catastrophic forgetting occur?

**Answer:** Stage 2 JSON fine-tuning achieved perfect structured output validity scores (5.0/5) but caused measurable forgetting in ROUGE metrics (-30%). BERTScore held (−1.4%), suggesting semantic meaning was preserved despite surface-level regression.

---

## Pipeline

```
Base Phi-3.5 Mini (Checkpoint 0)
         │
         ▼
Stage 1: QLoRA on Alpaca-Cleaned (46,584 examples)
         │
         ▼
Checkpoint 1: Alpaca-tuned model
         │
         ▼
Stage 2: QLoRA on Teacher-Generated JSON (112 examples × 10 epochs)
         │
         ▼
Checkpoint 2: JSON-tuned model
```

---

## Repository Structure

```
instruction-tuning-forgetting-analysis/
├── config.yaml                  # All hyperparameters
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── REPORT.md                    # GitHub blog post with full results
│
├── data/
│   ├── alpaca_train.json        # Stage 1 training (46,584 examples)
│   ├── alpaca_eval.json         # Alpaca held-out eval (5,176 examples)
│   ├── json_train.json          # Stage 2 training (112 examples)
│   ├── json_eval.json           # JSON held-out eval (13 examples)
│   └── json_eval_full.json      # Full JSON eval set (125 examples)
│
├── prompts/
│   ├── json_extraction.txt      # Teacher prompt: JSON extraction
│   ├── schema_generation.txt    # Teacher prompt: Schema generation
│   ├── classification.txt       # Teacher prompt: Classification
│   ├── json_repair.txt          # Teacher prompt: JSON repair
│   ├── tool_call.txt            # Teacher prompt: Tool-call generation
│   ├── judge_alpaca.txt         # Judge prompt: Alpaca evaluation
│   └── judge_json.txt           # Judge prompt: JSON evaluation
│
├── scripts/
│   ├── slurm_stage1.sh          # UTSA HPC SLURM batch script Stage 1
│   └── slurm_stage2.sh          # UTSA HPC SLURM batch script Stage 2
│
├── prepare_alpaca.py            # Stage 1 data preparation
├── generate_teacher_data.py     # Stage 2 imitation learning pipeline
├── train_stage1.py              # Stage 1 QLoRA training
├── train_stage2.py              # Stage 2 QLoRA training
├── inference.py                 # Generate outputs at each checkpoint
├── judge_eval.py                # LLM-as-a-Judge pairwise evaluation
├── compute_metrics.py           # Automatic metrics computation
├── ablation.py                  # Ablation study runner
│
├── results/
│   ├── checkpoint0_alpaca.json  # Checkpoint 0 Alpaca outputs
│   ├── checkpoint0_json.json    # Checkpoint 0 JSON outputs
│   ├── checkpoint1_alpaca.json  # Checkpoint 1 Alpaca outputs
│   ├── checkpoint1_json.json    # Checkpoint 1 JSON outputs
│   ├── checkpoint2_alpaca.json  # Checkpoint 2 Alpaca outputs
│   ├── checkpoint2_json.json    # Checkpoint 2 JSON outputs
│   ├── judge_*.json             # Judge evaluation results
│   ├── all_metrics.json         # All computed metrics
│   └── ablations/               # Ablation study results
```

---

## Setup Instructions

### Prerequisites
- UTSA ARC HPC access
- GPU node (V100 32GB recommended)
- UTSA VPN (for Llama 3.3 70B access)
- Python 3.10

### Step 1: Connect to UTSA ARC

```bash
ssh fpb170@arc.utsa.edu
```

### Step 2: Request a GPU node

Go to portal.arc.utsa.edu → My Interactive Sessions → Arc Virtual Desktop → Select GPU node (gpu1v100, 72 hours)

### Step 3: Set up Conda environment

```bash
module load anaconda3
conda create -p /work/fpb170/.conda/envs/llm python=3.10 -y
conda activate /work/fpb170/.conda/envs/llm
pip install -r requirements.txt
```

### Step 4: Set environment variables

Add to ~/.bashrc:
```bash
export CONDA_ENVS_DIRS=/work/fpb170/.conda/envs
export HF_HOME=/work/fpb170/hf_cache
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
export WANDB_PROJECT=assignment3
export UTSA_API_KEY_70B=your_70b_api_key
export UTSA_BASE_URL_70B=http://10.246.100.230/v1
export UTSA_MODEL_70B=llama-3.3-70b-instruct-awq
```

---

## Reproduction Steps

```bash
cd /work/fpb170/assignment3

# Step 1: Prepare Alpaca data
python prepare_alpaca.py

# Step 2: Generate teacher JSON data (connect UTSA VPN first)
python generate_teacher_data.py

# Step 3: Checkpoint 0 inference (baseline)
python inference.py --checkpoint checkpoint0

# Step 4: Train Stage 1
python train_stage1.py

# Step 5: Checkpoint 1 inference
python inference.py --checkpoint checkpoint1

# Step 6: Train Stage 2
python train_stage2.py

# Step 7: Checkpoint 2 inference
python inference.py --checkpoint checkpoint2

# Step 8: Judge evaluation (connect UTSA VPN first)
python judge_eval.py --comparison all

# Step 9: Compute metrics
python compute_metrics.py

# Step 10: Ablation study
python ablation.py --ablation epochs
```

### Using SLURM

```bash
sbatch scripts/slurm_stage1.sh
squeue -u fpb170
sbatch scripts/slurm_stage2.sh
```

---

## Model Configuration

| Parameter | Value |
|---|---|
| Student model | microsoft/Phi-3.5-mini-instruct |
| Teacher model | Llama-3.3-70B-Instruct-awq (UTSA hosted) |
| Judge model | Llama-3.3-70B-Instruct-awq (UTSA hosted) |
| Fine-tuning method | QLoRA (4-bit NF4) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Stage 1 epochs | 2 |
| Stage 1 batch size | 4 |
| Stage 1 max seq length | 1024 |
| Stage 1 learning rate | 2e-5 |
| Stage 2 epochs | 10 |
| Stage 2 batch size | 4 |
| Stage 2 max seq length | 1024 |
| Stage 2 learning rate | 2e-5 |

---

## Dataset Statistics

| Dataset | Examples |
|---|---|
| Alpaca train | 46,584 |
| Alpaca eval | 100 (used for evaluation) |
| Teacher JSON total | 125 (25 per task type) |
| Teacher JSON train | 112 |
| Teacher JSON eval | 13 |

**Teacher JSON task types (25 examples each):**
- JSON Extraction from unstructured text
- Schema-Constrained Generation
- Exact-Label Classification with JSON output
- JSON Repair / Formatting Correction
- Tool-Call Argument Generation

---

## Key Results

### Three-Checkpoint Comparison

| Checkpoint | ROUGE-L | BERTScore | JSON Validity |
|---|---|---|---|
| Checkpoint 0 (base) | 0.2032 | 0.8573 | 100% |
| Checkpoint 1 (Alpaca) | 0.1587 | 0.8408 | 100% |
| Checkpoint 2 (JSON) | 0.1105 | 0.8292 | 100% |

### Judge Win Rates (Alpaca)

| Comparison | Winner | Win Rate |
|---|---|---|
| C0 vs C1 | Checkpoint 0 | 82.0% |
| C1 vs C2 | Checkpoint 2 | 47.0% |
| C0 vs C2 | Checkpoint 0 | 73.0% |

### Forgetting Analysis

| Metric | C1 to C2 Change | Verdict |
|---|---|---|
| ROUGE-L | -30.4% | FORGETTING |
| BERTScore | -1.4% | MAINTAINED |
| Judge Win Rate | C2 wins 47% vs C1 26% | IMPROVED |

---

## UTSA HPC Details

- **Cluster:** UTSA ARC
- **Node:** Tesla V100S-PCIE-32GB (32GB VRAM)
- **CUDA:** 12.3
- **Stage 1 training time:** ~4 hours
- **Stage 2 training time:** ~6 minutes

---

## References

1. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
2. Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
3. Taori et al. (2023). Alpaca: A Strong, Replicable Instruction-Following Model.
4. Wang et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions.
5. Gu et al. (2024). A Survey on LLM-as-a-Judge.
