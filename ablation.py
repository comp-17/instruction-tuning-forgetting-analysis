"""
Assignment 3 - Ablation Study
Runs 3 ablation experiments on Stage 2 training:

Ablation 1: Vary Stage 2 epochs (1 vs 2 vs 3)
Ablation 2: Vary Stage 2 learning rate (2e-5 vs 1e-5 vs 5e-6)
Ablation 3: Vary Stage 2 dataset size (100% vs 50% vs 25%)

Each ablation variant:
1. Trains Stage 2 with the varied parameter
2. Runs inference on both Alpaca and JSON eval sets
3. Computes metrics
4. Saves results for comparison

Usage:
    python ablation.py --ablation epochs
    python ablation.py --ablation learning_rate
    python ablation.py --ablation dataset_size
    python ablation.py --ablation all
"""

import os
import json
import yaml
import argparse
import random
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from rouge_score import rouge_scorer

# ── Load Config ────────────────────────────────────────────────────────────────
CONFIG_PATH = "/work/fpb170/assignment3/config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_NAME     = config["model"]["student"]
LORA_CFG       = config["lora"]
PATHS          = config["paths"]
BASE_DIR       = PATHS["base_dir"]
STAGE1_ADAPTER = os.path.join(BASE_DIR, config["stage2"]["stage1_checkpoint"])
TRAIN_DATA     = os.path.join(BASE_DIR, config["stage2"]["train_file"])
EVAL_DATA      = os.path.join(BASE_DIR, config["stage2"]["eval_file"])
ALPACA_EVAL    = os.path.join(BASE_DIR, config["stage1"]["eval_file"])
ABLATION_DIR   = os.path.join(BASE_DIR, "outputs/ablations")
RESULTS_DIR    = os.path.join(BASE_DIR, "results/ablations")
RANDOM_SEED    = 42

# ── Ablation Variants ──────────────────────────────────────────────────────────
ABLATION_EPOCHS = [
    {"name": "epochs_1", "epochs": 1, "lr": 2e-5, "data_pct": 1.0},
    {"name": "epochs_2", "epochs": 2, "lr": 2e-5, "data_pct": 1.0},  # baseline
    {"name": "epochs_3", "epochs": 3, "lr": 2e-5, "data_pct": 1.0},
]

ABLATION_LR = [
    {"name": "lr_2e5",  "epochs": 2, "lr": 2e-5, "data_pct": 1.0},  # baseline
    {"name": "lr_1e5",  "epochs": 2, "lr": 1e-5, "data_pct": 1.0},
    {"name": "lr_5e6",  "epochs": 2, "lr": 5e-6, "data_pct": 1.0},
]

ABLATION_DATA = [
    {"name": "data_100", "epochs": 2, "lr": 2e-5, "data_pct": 1.00},  # baseline
    {"name": "data_50",  "epochs": 2, "lr": 2e-5, "data_pct": 0.50},
    {"name": "data_25",  "epochs": 2, "lr": 2e-5, "data_pct": 0.25},
]
# ───────────────────────────────────────────────────────────────────────────────


def format_prompt(example, include_output=True):
    """Same instruction template used throughout the pipeline."""
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


def load_json_data(data_pct=1.0):
    """Load teacher JSON training data with optional size reduction."""
    with open(TRAIN_DATA) as f:
        train_data = json.load(f)
    with open(EVAL_DATA) as f:
        eval_data = json.load(f)

    # Reduce dataset size for ablation
    if data_pct < 1.0:
        random.seed(RANDOM_SEED)
        random.shuffle(train_data)
        n = max(1, int(len(train_data) * data_pct))
        train_data = train_data[:n]

    train_formatted = [{"text": format_prompt(ex)} for ex in train_data]
    eval_formatted  = [{"text": format_prompt(ex)} for ex in eval_data]

    return (
        Dataset.from_list(train_formatted),
        Dataset.from_list(eval_formatted),
        train_data,
        eval_data,
    )


def load_alpaca_eval():
    """Load Alpaca eval set (100 examples)."""
    with open(ALPACA_EVAL) as f:
        data = json.load(f)
    return data[:100]


def load_base_model_with_stage1():
    """Load base model + Stage 1 adapter."""
    print(f"\n[Model] Loading base model + Stage 1 adapter...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

    model = PeftModel.from_pretrained(
        model, STAGE1_ADAPTER, is_trainable=True
    )

    return model, tokenizer


def train_ablation_variant(variant, train_dataset, eval_dataset):
    """Train a single ablation variant of Stage 2."""
    name       = variant["name"]
    epochs     = variant["epochs"]
    lr         = variant["lr"]
    output_dir = os.path.join(ABLATION_DIR, name)

    print(f"\n[Train] Ablation variant: {name}")
    print(f"        Epochs:   {epochs}")
    print(f"        LR:       {lr}")
    print(f"        Data:     {int(variant['data_pct']*100)}%")
    print(f"        Train:    {len(train_dataset)} examples")

    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_base_model_with_stage1()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="none",
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
        max_seq_length=1024,
        dataset_text_field="text",
        packing=False,
    )

    trainer.train()

    # Save final adapter
    final_path = os.path.join(output_dir, "final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"[Train] ✅ {name} complete → {final_path}")

    # Free memory
    del model
    torch.cuda.empty_cache()

    return final_path


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """Generate a response for a single prompt."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def run_inference_for_variant(adapter_path, variant_name, alpaca_data, json_data):
    """Run inference for an ablation variant on both eval sets."""
    print(f"\n[Inference] Running inference for {variant_name}...")

    # Load model with ablation adapter
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    # Run on Alpaca eval
    alpaca_results = []
    for i, ex in enumerate(tqdm(alpaca_data, desc=f"{variant_name}/alpaca")):
        prompt   = format_prompt(ex, include_output=False)
        response = generate_response(model, tokenizer, prompt)
        alpaca_results.append({
            "id":          i,
            "instruction": ex["instruction"],
            "input":       ex.get("input", ""),
            "reference":   ex.get("output", ""),
            "response":    response,
        })

    # Run on JSON eval
    with open(EVAL_DATA) as f:
        json_eval = json.load(f)

    json_results = []
    for i, ex in enumerate(tqdm(json_eval, desc=f"{variant_name}/json")):
        prompt   = format_prompt(ex, include_output=False)
        response = generate_response(model, tokenizer, prompt)
        json_results.append({
            "id":          i,
            "instruction": ex["instruction"],
            "input":       ex.get("input", ""),
            "reference":   ex.get("output", ""),
            "response":    response,
            "task_type":   ex.get("task_type", ""),
        })

    # Free memory
    del model
    torch.cuda.empty_cache()

    return alpaca_results, json_results


def compute_quick_metrics(alpaca_results, json_results):
    """Compute key metrics for ablation comparison."""
    # ROUGE-L for Alpaca
    scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl_scores = []
    for ex in alpaca_results:
        if ex["reference"] and ex["response"]:
            s = scorer.score(ex["reference"], ex["response"])
            rl_scores.append(s["rougeL"].fmeasure)

    rouge_l = round(sum(rl_scores) / len(rl_scores), 4) if rl_scores else 0

    # JSON validity
    valid = 0
    for ex in json_results:
        response = ex["response"].strip()
        if response.startswith("```"):
            lines    = response.split("\n")
            response = "\n".join(lines[1:-1])
        try:
            json.loads(response)
            valid += 1
        except:
            pass

    json_validity = round(valid / len(json_results), 4) if json_results else 0

    # Exact match for JSON
    matches = 0
    for ex in json_results:
        response  = ex["response"].strip()
        reference = ex["reference"].strip()
        if response.startswith("```"):
            lines    = response.split("\n")
            response = "\n".join(lines[1:-1])
        try:
            if json.loads(response) == json.loads(reference):
                matches += 1
        except:
            if response == reference:
                matches += 1

    exact_match = round(matches / len(json_results), 4) if json_results else 0

    return {
        "rouge_l":       rouge_l,
        "json_validity": json_validity,
        "exact_match":   exact_match,
    }


def save_ablation_results(variant_name, variant_config, metrics,
                          alpaca_results, json_results):
    """Save ablation results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = {
        "variant":        variant_name,
        "config":         variant_config,
        "metrics":        metrics,
        "alpaca_results": alpaca_results,
        "json_results":   json_results,
    }

    path = os.path.join(RESULTS_DIR, f"ablation_{variant_name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Save] ✅ Saved to {path}")
    return path


def run_ablation(variants, ablation_name):
    """Run a full ablation study across all variants."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_name}")
    print(f"Variants: {[v['name'] for v in variants]}")
    print(f"{'='*60}")

    # Check Stage 1 adapter exists
    if not os.path.exists(STAGE1_ADAPTER):
        print(f"❌ ERROR: Stage 1 adapter not found at {STAGE1_ADAPTER}")
        print(f"   Run train_stage1.py first!")
        return

    alpaca_data = load_alpaca_eval()
    all_results = []

    for variant in variants:
        name     = variant["name"]
        data_pct = variant["data_pct"]

        print(f"\n{'─'*50}")
        print(f"Running variant: {name}")
        print(f"{'─'*50}")

        # Load data
        train_dataset, eval_dataset, _, _ = load_json_data(data_pct)

        # Train
        adapter_path = train_ablation_variant(
            variant, train_dataset, eval_dataset
        )

        # Inference
        alpaca_results, json_results = run_inference_for_variant(
            adapter_path, name, alpaca_data, []
        )

        # Metrics
        metrics = compute_quick_metrics(alpaca_results, json_results)

        print(f"\n[Results] {name}:")
        print(f"  ROUGE-L:       {metrics['rouge_l']}")
        print(f"  JSON validity: {metrics['json_validity']}")
        print(f"  Exact match:   {metrics['exact_match']}")

        # Save
        save_ablation_results(
            name, variant, metrics, alpaca_results, json_results
        )

        all_results.append({
            "variant": name,
            "config":  variant,
            "metrics": metrics,
        })

    # Save combined ablation summary
    summary_path = os.path.join(RESULTS_DIR, f"ablation_{ablation_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"ABLATION RESULTS: {ablation_name}")
    print(f"{'='*60}")
    print(f"\n{'Variant':<15} {'ROUGE-L':<12} {'JSON Valid':<12} {'Exact Match'}")
    print(f"{'─'*55}")
    for r in all_results:
        m = r["metrics"]
        print(
            f"{r['variant']:<15} "
            f"{m['rouge_l']:<12} "
            f"{m['json_validity']:<12} "
            f"{m['exact_match']}"
        )

    print(f"\n✅ {ablation_name} ablation complete!")
    print(f"   Summary → {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation studies for Assignment 3"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=["epochs", "learning_rate", "dataset_size", "all"],
        help="Which ablation to run"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Assignment 3 - Ablation Study")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU found!")
        return

    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    os.makedirs(ABLATION_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,  exist_ok=True)

    if args.ablation == "epochs" or args.ablation == "all":
        run_ablation(ABLATION_EPOCHS, "epochs")

    if args.ablation == "learning_rate" or args.ablation == "all":
        run_ablation(ABLATION_LR, "learning_rate")

    if args.ablation == "dataset_size" or args.ablation == "all":
        run_ablation(ABLATION_DATA, "dataset_size")

    print(f"\n{'='*60}")
    print("All Ablations Complete!")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
