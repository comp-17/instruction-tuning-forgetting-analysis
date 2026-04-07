# Assignment 3: Sequential Instruction Tuning of a Small LLM

**Course:** LLM & Agentic Systems — Graduate Course  
**Student:** fpb170  
**Due:** April 6th, 2026  

## Overview

This repository implements a two-stage instruction-tuning pipeline for a small language model (Phi-3.5 Mini Instruct), evaluating the model at three checkpoints using both automatic metrics and a strong LLM judge (Llama 3.3 70B).

**Core Research Question:** Does fine-tuning on Alpaca data followed by teacher-generated JSON data improve structured-output reliability while maintaining general instruction-following ability, or does catastrophic forgetting occur?

---

## Pipeline Overview

```
Stage 1: Alpaca Fine-Tuning          Stage 2: Teacher JSON Fine-Tuning
─────────────────────────            ──────────────────────────────────
Base Phi-3.5 Mini                    Stage 1 Checkpoint
      │                                     │
      ▼                                     ▼
QLoRA on Alpaca Data           QLoRA on Teacher-Generated JSON
      │                                     │
      ▼                                     ▼
Checkpoint 1                         Checkpoint 2
```

**Three Checkpoints Evaluated:**
- Checkpoint 0: Untuned base model (baseline)
- Checkpoint 1: After Stage 1 Alpaca fine-tuning
- Checkpoint 2: After Stage 2 Teacher JSON fine-tuning

---

## Repository Structure

```
assignment3/
├── config.yaml                  # All hyperparameters
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── REPORT.md                    # GitHub blog post
│
├── data/
│   ├── alpaca_train.json        # Stage 1 training data
│   ├── alpaca_eval.json         # Alpaca held-out eval set
│   ├── json_train.json          # Stage 2 training data
│   └── json_eval.json           # JSON held-out eval set
│
├── prompts/
│   ├── json_extraction.txt      # Teacher prompt template
│   ├── schema_generation.txt    # Teacher prompt template
│   ├── classification.txt       # Teacher prompt template
│   ├── json_repair.txt          # Teacher prompt template
│   ├── tool_call.txt            # Teacher prompt template
│   ├── judge_alpaca.txt         # Judge prompt for Alpaca eval
│   └── judge_json.txt           # Judge prompt for JSON eval
│
├── scripts/
│   ├── slurm_stage1.sh          # UTSA HPC batch script Stage 1
│   └── slurm_stage2.sh          # UTSA HPC batch script Stage 2
│
├── prepare_alpaca.py            # Stage 1 data preparation
├── generate_teacher_data.py     # Stage 2 data generation (imitation learning)
├── generate_missing.py          # Fix missing examples
├── train_stage1.py              # Stage 1 QLoRA training
├── train_stage2.py              # Stage 2 QLoRA training
├── inference.py                 # Generate outputs at each checkpoint
├── judge_eval.py                # LLM judge evaluation
├── compute_metrics.py           # Automatic metrics computation
│
├── outputs/
│   ├── stage1/final/            # Stage 1 LoRA adapter
│   └── stage2/final/            # Stage 2 LoRA adapter
│
└── results/
    ├── checkpoint0_alpaca.json  # Checkpoint 0 Alpaca outputs
    ├── checkpoint0_json.json    # Checkpoint 0 JSON outputs
    ├── checkpoint1_alpaca.json  # Checkpoint 1 Alpaca outputs
    ├── checkpoint1_json.json    # Checkpoint 1 JSON outputs
    ├── checkpoint2_alpaca.json  # Checkpoint 2 Alpaca outputs
    ├── checkpoint2_json.json    # Checkpoint 2 JSON outputs
    ├── judge_*.json             # Judge evaluation results
    └── all_metrics.json         # All computed metrics
```

---

## Setup Instructions

### Prerequisites
- UTSA ARC HPC access
- GPU node (V100 or A100 recommended)
- UTSA VPN (for Llama 3.3 70B access)

### Step 1: Connect to UTSA ARC

```bash
ssh fpb170@arc.utsa.edu
```

### Step 2: Request a GPU node

Go to `portal.arc.utsa.edu` → My Interactive Sessions → Arc Virtual Desktop → Select GPU node

### Step 3: Set up environment

```bash
# Load anaconda
module load anaconda3

# Create conda environment
conda create -p /work/fpb170/.conda/envs/llm python=3.10 -y
conda activate /work/fpb170/.conda/envs/llm

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Set environment variables

Add to `~/.bashrc`:
```bash
export CONDA_ENVS_DIRS=/work/fpb170/.conda/envs
export HF_HOME=/work/fpb170/hf_cache
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
export WANDB_PROJECT=assignment3

export UTSA_API_KEY_8B=utsa-jABQlGLaTrae2bqMHyAvPxTvE9KTP0DEWYIXhvtgkDkVcGjp44rN6G56x1aGiyem
export UTSA_BASE_URL_8B=http://149.165.173.247:8888/v1
export UTSA_MODEL_8B=meta-llama/Llama-3.1-8B-Instruct

export UTSA_API_KEY_70B=gpustack_50e00c9281422bc5_0c0696dfcb1696d7635e58a2e56d6282
export UTSA_BASE_URL_70B=http://10.246.100.230/v1
export UTSA_MODEL_70B=llama-3.3-70b-instruct-awq
```

---

## Reproduction Steps

Run these commands in order to reproduce all results:

```bash
cd /work/fpb170/assignment3

# Step 1: Prepare Alpaca data (Stage 1)
python prepare_alpaca.py

# Step 2: Generate teacher JSON data (Stage 2)
# Connect to UTSA VPN first for 70B model access
python generate_teacher_data.py

# Step 3: Run Checkpoint 0 inference (baseline)
python inference.py --checkpoint checkpoint0

# Step 4: Train Stage 1 (Alpaca fine-tuning)
python train_stage1.py

# Step 5: Run Checkpoint 1 inference
python inference.py --checkpoint checkpoint1

# Step 6: Train Stage 2 (Teacher JSON fine-tuning)
python train_stage2.py

# Step 7: Run Checkpoint 2 inference
python inference.py --checkpoint checkpoint2

# Step 8: Run judge evaluation
python judge_eval.py --comparison all

# Step 9: Compute all metrics
python compute_metrics.py
```

### Using SLURM (batch jobs)

```bash
# Submit Stage 1 training as batch job
sbatch scripts/slurm_stage1.sh

# Check job status
squeue -u fpb170

# Submit Stage 2 after Stage 1 completes
sbatch scripts/slurm_stage2.sh
```

---

## Model Configuration

| Parameter | Value |
|---|---|
| Student model | microsoft/Phi-3.5-mini-instruct |
| Teacher model | Llama-3.3-70B-Instruct-awq |
| Judge model | Llama-3.3-70B-Instruct-awq |
| Fine-tuning method | QLoRA (4-bit) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 2e-5 |
| Stage 1 epochs | 2 |
| Stage 2 epochs | 2 |
| Max sequence length | 1024 |
| Batch size | 4 |
| Gradient accumulation | 4 |

---

## Dataset Statistics

| Dataset | Examples |
|---|---|
| Alpaca train | 46,584 |
| Alpaca eval | 5,176 (100 used for evaluation) |
| Teacher JSON train | 112 |
| Teacher JSON eval | 13 |

**Teacher JSON task breakdown:**
- JSON Extraction: 25 examples
- Schema-Constrained Generation: 25 examples
- Classification with JSON output: 25 examples
- JSON Repair: 25 examples
- Tool-Call Argument Generation: 25 examples

---

## Results

See `results/all_metrics.json` for complete results and `REPORT.md` for analysis.

---

## References

1. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
2. Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
3. Taori et al. (2023). Alpaca: A Strong, Replicable Instruction-Following Model.
4. Wang et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions.
5. Gu et al. (2024). A Survey on LLM-as-a-Judge.

---

## Ablation Studies

We run 3 ablation experiments on Stage 2 training:

### Ablation 1: Vary Stage 2 Epochs
| Variant | Epochs | LR | Data |
|---|---|---|---|
| epochs_1 | 1 | 2e-5 | 100% |
| epochs_2 | 2 | 2e-5 | 100% (baseline) |
| epochs_3 | 3 | 2e-5 | 100% |

### Ablation 2: Vary Stage 2 Learning Rate
| Variant | Epochs | LR | Data |
|---|---|---|---|
| lr_2e5 | 2 | 2e-5 | 100% (baseline) |
| lr_1e5 | 2 | 1e-5 | 100% |
| lr_5e6 | 2 | 5e-6 | 100% |

### Ablation 3: Vary Stage 2 Dataset Size
| Variant | Epochs | LR | Data |
|---|---|---|---|
| data_100 | 2 | 2e-5 | 100% (baseline) |
| data_50  | 2 | 2e-5 | 50% |
| data_25  | 2 | 2e-5 | 25% |

### Running Ablations
```bash
# Run all ablations (after Stage 1 training completes)
python ablation.py --ablation epochs
python ablation.py --ablation learning_rate
python ablation.py --ablation dataset_size
```

Results saved to: `results/ablations/`
