"""
Assignment 3 - Stage 2 Training
Continues fine-tuning from Stage 1 checkpoint on teacher-generated JSON data.

Assignment Specifications:
- Loads Stage 1 adapter as starting point
- Trains on teacher-generated JSON Instruct dataset
- Same QLoRA configuration as Stage 1
- Learning rate: 2e-5, Epochs: 2
"""

import os
import json
import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
import wandb

# ── Load Config ────────────────────────────────────────────────────────────────
CONFIG_PATH = "/work/fpb170/assignment3/config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_NAME      = config["model"]["student"]
STAGE2_CFG      = config["stage2"]
LORA_CFG        = config["lora"]
PATHS           = config["paths"]
WANDB_CFG       = config["wandb"]

OUTPUT_DIR      = os.path.join(PATHS["base_dir"], STAGE2_CFG["output_dir"])
TRAIN_DATA_PATH = os.path.join(PATHS["base_dir"], STAGE2_CFG["train_file"])
EVAL_DATA_PATH  = os.path.join(PATHS["base_dir"], STAGE2_CFG["eval_file"])
STAGE1_ADAPTER  = os.path.join(PATHS["base_dir"], STAGE2_CFG["stage1_checkpoint"])
# ───────────────────────────────────────────────────────────────────────────────


def format_prompt(example, include_output=True):
    """
    Same instruction template as Stage 1.
    MUST be consistent across all stages and inference.
    """
    instruction = example["instruction"]
    input_text  = example.get("input", "")
    output      = example.get("output", "")

    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    if include_output:
        return prompt + output
    return prompt


def load_data():
    """Load and format the teacher-generated JSON dataset."""
    print(f"\n[Data] Loading training data from {TRAIN_DATA_PATH}")
    with open(TRAIN_DATA_PATH) as f:
        train_data = json.load(f)

    print(f"[Data] Loading eval data from {EVAL_DATA_PATH}")
    with open(EVAL_DATA_PATH) as f:
        eval_data = json.load(f)

    # Format into text field
    train_formatted = [{"text": format_prompt(ex)} for ex in train_data]
    eval_formatted  = [{"text": format_prompt(ex)} for ex in eval_data]

    train_dataset = Dataset.from_list(train_formatted)
    eval_dataset  = Dataset.from_list(eval_formatted)

    print(f"[Data] Train set: {len(train_dataset)} examples")
    print(f"[Data] Eval set:  {len(eval_dataset)} examples")

    # Print task type breakdown
    task_counts = {}
    for ex in train_data + eval_data:
        t = ex.get("task_type", "unknown")
        task_counts[t] = task_counts.get(t, 0) + 1
    print(f"[Data] Task breakdown:")
    for task, count in sorted(task_counts.items()):
        print(f"         {task}: {count}")

    return train_dataset, eval_dataset


def load_model_and_tokenizer():
    """
    Load Phi-3.5 Mini in 4-bit and apply Stage 1 LoRA adapter.
    This is the key difference from Stage 1 training.
    """
    print(f"\n[Model] Loading {MODEL_NAME} in 4-bit...")
    print(f"[Model] Loading Stage 1 adapter from {STAGE1_ADAPTER}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Load Stage 1 LoRA adapter on top of base model
    model = PeftModel.from_pretrained(
        model,
        STAGE1_ADAPTER,
        is_trainable=True,
    )

    trainable, total = model.get_nb_trainable_parameters()
    print(f"[Model] Stage 1 adapter loaded!")
    print(f"[Model] Trainable: {trainable:,} ({100*trainable/total:.2f}% of {total:,} total)")

    return model, tokenizer


def train(model, tokenizer, train_dataset, eval_dataset):
    """Run Stage 2 QLoRA training on teacher-generated JSON data."""
    print(f"\n[Train] Starting Stage 2 training...")
    print(f"        Starting from: Stage 1 checkpoint")
    print(f"        Dataset:       Teacher-generated JSON")
    print(f"        Epochs:        {STAGE2_CFG['num_epochs']}")
    print(f"        Learning rate: {STAGE2_CFG['learning_rate']}")
    print(f"        Batch size:    {STAGE2_CFG['batch_size']}")
    print(f"        Output:        {OUTPUT_DIR}")

    # Initialize W&B
    wandb.init(
        project=WANDB_CFG["project"],
        name=WANDB_CFG["stage2_run"],
        config={
            "model":          MODEL_NAME,
            "stage":          "stage2",
            "dataset":        "teacher_json",
            "stage1_adapter": STAGE1_ADAPTER,
            "lora_rank":      LORA_CFG["rank"],
            "lora_alpha":     LORA_CFG["alpha"],
            "lora_dropout":   LORA_CFG["dropout"],
            "learning_rate":  STAGE2_CFG["learning_rate"],
            "epochs":         STAGE2_CFG["num_epochs"],
            "batch_size":     STAGE2_CFG["batch_size"],
            "max_seq_length": STAGE2_CFG["max_seq_length"],
        }
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=STAGE2_CFG["num_epochs"],
        per_device_train_batch_size=STAGE2_CFG["batch_size"],
        per_device_eval_batch_size=STAGE2_CFG["batch_size"],
        gradient_accumulation_steps=STAGE2_CFG["gradient_accumulation_steps"],
        learning_rate=STAGE2_CFG["learning_rate"],
        warmup_ratio=STAGE2_CFG["warmup_ratio"],
        lr_scheduler_type=STAGE2_CFG["lr_scheduler"],
        fp16=True,
        logging_steps=STAGE2_CFG["logging_steps"],
        save_steps=STAGE2_CFG["save_steps"],
        eval_steps=STAGE2_CFG["save_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=WANDB_CFG["stage2_run"],
        dataloader_num_workers=0,
        remove_unused_columns=True,
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=STAGE2_CFG["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    print("\n[Train] Training started! Monitor at wandb.ai")
    print("[Train] Saving checkpoint every 50 steps...")
    print("[Train] This will take 1-2 hours...\n")

    trainer.train()

    # Save final adapter checkpoint
    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save training summary
    summary = {
        "stage": "stage2",
        "model": MODEL_NAME,
        "dataset": "teacher_json",
        "stage1_adapter": STAGE1_ADAPTER,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "epochs": STAGE2_CFG["num_epochs"],
        "learning_rate": STAGE2_CFG["learning_rate"],
        "lora_rank": LORA_CFG["rank"],
        "lora_alpha": LORA_CFG["alpha"],
        "final_checkpoint": final_path,
    }

    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Train] ✅ Stage 2 training complete!")
    print(f"[Train] Final adapter: {final_path}")
    print(f"[Train] Summary:       {summary_path}")

    wandb.finish()
    return final_path


def main():
    print("=" * 60)
    print("Assignment 3 - Stage 2 Training")
    print("Phi-3.5 Mini + Stage1 Adapter + QLoRA on Teacher JSON")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU found!")
        return

    gpu_name   = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n[GPU] {gpu_name}")
    print(f"[GPU] Memory: {gpu_memory:.1f} GB")

    # Check Stage 1 adapter exists
    if not os.path.exists(STAGE1_ADAPTER):
        print(f"❌ ERROR: Stage 1 adapter not found at {STAGE1_ADAPTER}")
        print(f"   Run train_stage1.py first!")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    train_dataset, eval_dataset = load_data()

    # Load model + Stage 1 adapter
    model, tokenizer = load_model_and_tokenizer()

    # Train
    final_path = train(model, tokenizer, train_dataset, eval_dataset)

    print(f"\n{'='*60}")
    print("Stage 2 Complete!")
    print(f"Checkpoint: {final_path}")
    print(f"Next step:  Run inference.py --checkpoint checkpoint2")
    print(f"            Then run judge_eval.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
