"""
Combined Training Ablation: Single-stage training on merged Alpaca + Teacher JSON data.
Compares against sequential two-stage training pipeline.
"""

import os
import json
import random
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

# ── Config ─────────────────────────────────────────────────────
BASE_DIR      = "/work/fpb170/assignment3"
MODEL_NAME    = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs/combined")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
DATA_DIR      = os.path.join(BASE_DIR, "data")

LORA_RANK     = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
NUM_EPOCHS    = 2
LEARNING_RATE = 2e-5
BATCH_SIZE    = 4
GRAD_ACCUM    = 4
MAX_SEQ_LEN   = 1024
SEED          = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def format_example(ex):
    """Format example into instruction template."""
    instruction = ex.get("instruction", "").strip()
    input_text  = ex.get("input", "").strip()
    output      = ex.get("output", "").strip()
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

def load_data():
    """Load and merge Alpaca + JSON training data."""
    # Load Alpaca train
    with open(os.path.join(DATA_DIR, "alpaca_train.json")) as f:
        alpaca_train = json.load(f)

    # Load JSON train
    with open(os.path.join(DATA_DIR, "json_train.json")) as f:
        json_train = json.load(f)

    # Load Alpaca eval (100 examples)
    with open(os.path.join(DATA_DIR, "alpaca_eval.json")) as f:
        alpaca_eval = json.load(f)[:100]

    # Merge train sets
    combined_train = alpaca_train + json_train
    random.seed(SEED)
    random.shuffle(combined_train)

    print(f"[Data] Alpaca train:   {len(alpaca_train):,} examples")
    print(f"[Data] JSON train:     {len(json_train):,} examples")
    print(f"[Data] Combined train: {len(combined_train):,} examples")
    print(f"[Data] Alpaca eval:    {len(alpaca_eval):,} examples")

    train_dataset = Dataset.from_list([{"text": format_example(ex)} for ex in combined_train])
    eval_dataset  = Dataset.from_list([{"text": format_example(ex)} for ex in alpaca_eval])

    return train_dataset, eval_dataset

def load_model():
    """Load Phi-3.5 Mini in 4-bit QLoRA."""
    print(f"[Model] Loading {MODEL_NAME} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/work/fpb170/hf_cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir="/work/fpb170/hf_cache",
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

def train(model, tokenizer, train_dataset, eval_dataset):
    """Run combined training."""
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="wandb",
        run_name="combined_alpaca_json",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args,
    )

    print("[Train] Starting combined training...")
    print(f"        Epochs:        {NUM_EPOCHS}")
    print(f"        Learning rate: {LEARNING_RATE}")
    print(f"        Batch size:    {BATCH_SIZE}")
    print(f"        Train examples:{len(train_dataset):,}")

    trainer.train(resume_from_checkpoint="/work/fpb170/assignment3/outputs/combined/checkpoint-5200")

    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[Train] ✅ Combined training complete!")
    print(f"[Train] Adapter saved to: {final_path}")

    # Save training summary
    summary = {
        "stage": "combined",
        "model": MODEL_NAME,
        "dataset": "alpaca_cleaned + teacher_json",
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "final_checkpoint": final_path
    }
    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return final_path

def main():
    print("=" * 60)
    print("Assignment 3 - Combined Training Ablation")
    print("Single-stage: Alpaca + Teacher JSON merged")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_dataset, eval_dataset = load_data()
    model, tokenizer = load_model()
    train(model, tokenizer, train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
