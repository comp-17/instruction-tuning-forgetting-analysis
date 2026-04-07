"""
Assignment 3 - Inference Script
Generates model outputs at all 3 checkpoints on both evaluation sets.

Checkpoints:
- Checkpoint 0: Untuned base model (baseline)
- Checkpoint 1: After Stage 1 Alpaca fine-tuning
- Checkpoint 2: After Stage 2 Teacher JSON fine-tuning

Evaluation sets:
- Alpaca eval: 100 general instruction prompts
- JSON eval: 100 structured output prompts (all 5 task types)

Usage:
    python inference.py --checkpoint checkpoint0
    python inference.py --checkpoint checkpoint1
    python inference.py --checkpoint checkpoint2
    python inference.py --checkpoint all
"""

import os
import json
import yaml
import argparse
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ── Load Config ────────────────────────────────────────────────────────────────
CONFIG_PATH = "/work/fpb170/assignment3/config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MODEL_NAME = config["model"]["student"]
PATHS      = config["paths"]
EVAL_CFG   = config["evaluation"]
BASE_DIR   = PATHS["base_dir"]
# ───────────────────────────────────────────────────────────────────────────────


def format_prompt(example, include_output=False):
    """
    Format example using the same instruction template as training.
    include_output=False for inference (we want model to generate response).
    """
    instruction = example["instruction"]
    input_text  = example.get("input", "")

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

    return prompt


def load_model(checkpoint_name, checkpoint_config):
    """
    Load model for a specific checkpoint.
    - Checkpoint 0: base model only (no adapter)
    - Checkpoint 1: base model + Stage 1 adapter
    - Checkpoint 2: base model + Stage 2 adapter
    """
    adapter_path = checkpoint_config.get("adapter")

    print(f"\n[Model] Loading {checkpoint_name}...")
    print(f"[Model] Base model: {MODEL_NAME}")
    if adapter_path:
        full_adapter_path = os.path.join(BASE_DIR, adapter_path)
        print(f"[Model] Adapter: {full_adapter_path}")
    else:
        print(f"[Model] No adapter (untuned base model)")

    # Load in float16 for inference (no quantization needed)
    bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    # Load adapter if specified
    if adapter_path:
        full_adapter_path = os.path.join(BASE_DIR, adapter_path)
        if not os.path.exists(full_adapter_path):
            print(f"❌ ERROR: Adapter not found at {full_adapter_path}")
            print(f"   Make sure training is complete for {checkpoint_name}")
            return None, None
        model = PeftModel.from_pretrained(model, full_adapter_path)
        model = model.merge_and_unload()  # Merge adapter for faster inference

    model.eval()
    print(f"[Model] ✅ {checkpoint_name} loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=1024):
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

    # Decode only the new tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def run_inference_on_dataset(model, tokenizer, data, dataset_name, checkpoint_name):
    """
    Run inference on a dataset and save results.
    Returns list of results with prompt, reference, and model output.
    """
    print(f"\n[Inference] Running {checkpoint_name} on {dataset_name}...")
    print(f"[Inference] Examples: {len(data)}")

    results = []

    for i, example in enumerate(tqdm(data, desc=f"{checkpoint_name}/{dataset_name}")):
        prompt    = format_prompt(example, include_output=False)
        reference = example.get("output", "")
        task_type = example.get("task_type", "general")

        try:
            response = generate_response(model, tokenizer, prompt)
        except Exception as e:
            print(f"\n⚠ Error on example {i}: {e}")
            response = ""

        results.append({
            "id":           i,
            "checkpoint":   checkpoint_name,
            "dataset":      dataset_name,
            "task_type":    task_type,
            "instruction":  example["instruction"],
            "input":        example.get("input", ""),
            "reference":    reference,
            "response":     response,
            "prompt":       prompt,
        })

    return results


def save_results(results, checkpoint_name, dataset_name):
    """Save inference results to results directory."""
    results_dir = PATHS["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    filename    = f"{checkpoint_name}_{dataset_name}.json"
    output_path = os.path.join(results_dir, filename)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Save] ✅ Saved {len(results)} results to {output_path}")
    return output_path


def run_checkpoint(checkpoint_name):
    """Run inference for a single checkpoint on both eval sets."""
    print(f"\n{'='*60}")
    print(f"Running inference for: {checkpoint_name}")
    print(f"{'='*60}")

    # Find checkpoint config
    checkpoint_config = None
    for ckpt in EVAL_CFG["checkpoints"]:
        if ckpt["name"] == checkpoint_name:
            checkpoint_config = ckpt
            break

    if checkpoint_config is None:
        print(f"❌ ERROR: Unknown checkpoint '{checkpoint_name}'")
        print(f"   Valid options: checkpoint0, checkpoint1, checkpoint2")
        return

    print(f"Description: {checkpoint_config['description']}")

    # Load model
    model, tokenizer = load_model(checkpoint_name, checkpoint_config)
    if model is None:
        return

    # ── Run on Alpaca eval set ─────────────────────────────────────────────────
    alpaca_eval_path = os.path.join(BASE_DIR, config["stage1"]["eval_file"])
    with open(alpaca_eval_path) as f:
        alpaca_data = json.load(f)

    # Use exactly 100 examples as required by assignment
    alpaca_data = alpaca_data[:EVAL_CFG["alpaca_eval_size"]]
    alpaca_results = run_inference_on_dataset(
        model, tokenizer, alpaca_data, "alpaca", checkpoint_name
    )
    save_results(alpaca_results, checkpoint_name, "alpaca")

    # ── Run on JSON eval set ───────────────────────────────────────────────────
    json_eval_path = os.path.join(BASE_DIR, 'data/json_eval_full.json')
    with open(json_eval_path) as f:
        json_data = json.load(f)

    json_results = run_inference_on_dataset(
        model, tokenizer, json_data, "json", checkpoint_name
    )
    save_results(json_results, checkpoint_name, "json")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n[Summary] {checkpoint_name} inference complete!")
    print(f"          Alpaca results: {len(alpaca_results)} examples")
    print(f"          JSON results:   {len(json_results)} examples")
    print(f"          Saved to:       {PATHS['results_dir']}/")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference at each checkpoint for Assignment 3"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        choices=["checkpoint0", "checkpoint1", "checkpoint2", "all"],
        help="Which checkpoint to run inference on"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Assignment 3 - Inference")
    print("Generating outputs at each checkpoint")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU found!")
        return

    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    if args.checkpoint == "all":
        for ckpt in ["checkpoint0", "checkpoint1", "checkpoint2"]:
            run_checkpoint(ckpt)
    else:
        run_checkpoint(args.checkpoint)

    print(f"\n{'='*60}")
    print("Inference Complete!")
    print(f"Results saved to: {PATHS['results_dir']}/")
    print(f"Next step: Run judge_eval.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
