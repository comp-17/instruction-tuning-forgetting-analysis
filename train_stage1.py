"""
Assignment 3 - Stage 1 Training
Fine-tunes Phi-3.5 Mini Instruct on Alpaca data using QLoRA.
Following assignment specifications:
- Model: microsoft/Phi-3.5-mini-instruct
- Method: QLoRA (4-bit quantization)
- LoRA rank: 16, alpha: 32, dropout: 0.05
- Learning rate: 2e-5
- Epochs: 2
- Max sequence length: 1024
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
import wandb

# ── Configuration (Assignment 3 Specifications) ────────────────────────────────
MODEL_NAME       = "microsoft/Phi-3.5-mini-instruct"
STAGE1_OUTPUT    = "/work/fpb170/assignment3/outputs/stage1"
TRAIN_DATA_PATH  = "/work/fpb170/assignment3/data/alpaca_train.json"
EVAL_DATA_PATH   = "/work/fpb170/assignment3/data/alpaca_eval.json"

# LoRA config (from assignment)
LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05

# Training config (from assignment)
LEARNING_RATE    = 2e-5
NUM_EPOCHS       = 2
MAX_SEQ_LENGTH   = 1024
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 4
SAVE_STEPS       = 50
LOGGING_STEPS    = 5
WARMUP_RATIO     = 0.03
# ───────────────────────────────────────────────────────────────────────────────


def format_prompt(example, include_output=True):
    """
    Format example using the instruction template.
    MUST be used consistently at training AND inference time.
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


def load_data(train_path, eval_path):
    """Load and format the Alpaca dataset."""
    print(f"\n[Data] Loading training data from {train_path}")
    with open(train_path) as f:
        train_data = json.load(f)

    print(f"[Data] Loading eval data from {eval_path}")
    with open(eval_path) as f:
        eval_data = json.load(f)

    # Use only 100 eval examples as per assignment requirement
    eval_data = eval_data[:100]

    # Format into text field
    train_formatted = [{"text": format_prompt(ex)} for ex in train_data]
    eval_formatted  = [{"text": format_prompt(ex)} for ex in eval_data]

    train_dataset = Dataset.from_list(train_formatted)
    eval_dataset  = Dataset.from_list(eval_formatted)

    print(f"[Data] Train set: {len(train_dataset)} examples")
    print(f"[Data] Eval set:  {len(eval_dataset)} examples")

    return train_dataset, eval_dataset


def load_model_and_tokenizer(model_name):
    """Load model in 4-bit quantization with QLoRA config."""
    print(f"\n[Model] Loading {model_name} in 4-bit...")

    # 4-bit quantization config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    print(f"[Model] Model loaded successfully!")
    return model, tokenizer


def apply_lora(model):
    """Apply LoRA adapters as per assignment specifications."""
    print(f"\n[LoRA] Applying LoRA adapters...")
    print(f"       Rank: {LORA_RANK}, Alpha: {LORA_ALPHA}, Dropout: {LORA_DROPOUT}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[LoRA] Trainable parameters: {trainable:,} ({100*trainable/total:.2f}% of {total:,} total)")

    return model


def train(model, tokenizer, train_dataset, eval_dataset):
    """Run Stage 1 training."""
    print(f"\n[Train] Starting Stage 1 training...")
    print(f"        Epochs: {NUM_EPOCHS}")
    print(f"        Learning rate: {LEARNING_RATE}")
    print(f"        Batch size: {BATCH_SIZE}")
    print(f"        Output: {STAGE1_OUTPUT}")

    # Initialize W&B
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "assignment3"),
        name="stage1_alpaca",
        config={
            "model": MODEL_NAME,
            "stage": "stage1",
            "dataset": "alpaca-cleaned",
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "max_seq_length": MAX_SEQ_LENGTH,
        }
    )

    training_args = TrainingArguments(
        output_dir=STAGE1_OUTPUT,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="wandb",
        run_name="stage1_alpaca",
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
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    print("\n[Train] Training started! Monitor at wandb.ai")
    print("[Train] This will take 2-4 hours...\n")

    trainer.train()

    # Save final adapter
    final_path = f"{STAGE1_OUTPUT}/final"
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\n[Train] ✅ Stage 1 training complete!")
    print(f"[Train] Final adapter saved to: {final_path}")

    wandb.finish()
    return final_path


def main():
    print("=" * 60)
    print("Assignment 3 - Stage 1 Training (Alpaca + QLoRA)")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU found! Make sure you are on a GPU node.")
        return

    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    train_dataset, eval_dataset = load_data(TRAIN_DATA_PATH, EVAL_DATA_PATH)

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Apply LoRA
    model = apply_lora(model)

    # Train
    final_path = train(model, tokenizer, train_dataset, eval_dataset)

    print(f"\n{'='*60}")
    print("Stage 1 Complete!")
    print(f"Checkpoint saved at: {final_path}")
    print(f"Next step: Run train_stage2.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
