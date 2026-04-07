"""
Assignment 3 - Stage 1 Data Preparation
Downloads and prepares the Alpaca-Cleaned dataset for instruction tuning.
Saves train and eval splits to /work/fpb170/assignment3/data/
"""

import json
import random
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_NAME    = "yahma/alpaca-cleaned"
OUTPUT_DIR      = "/work/fpb170/assignment3/data"
TRAIN_FILE      = f"{OUTPUT_DIR}/alpaca_train.json"
EVAL_FILE       = f"{OUTPUT_DIR}/alpaca_eval.json"
EVAL_SPLIT_SIZE = 0.1       # 10% held out for evaluation
RANDOM_SEED     = 42
# ───────────────────────────────────────────────────────────────────────────────


def format_example(example):
    """
    Format a raw Alpaca example into the standard schema:
    { instruction, input, output }
    Skips examples with empty instruction or output.
    """
    instruction = example.get("instruction", "").strip()
    input_text  = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    # Skip malformed examples
    if not instruction or not output:
        return None

    return {
        "instruction": instruction,
        "input":       input_text,
        "output":      output
    }


def apply_chat_template(example, include_output=True):
    """
    Apply the instruction template the model will be trained on.
    This MUST be used consistently at training AND inference time.

    Template format (following TA's recommendation):
    ### Instruction:
    {instruction}

    ### Input:
    {input}   <- only included if input is non-empty

    ### Response:
    {output}  <- only included during training
    """
    instruction = example["instruction"]
    input_text  = example["input"]
    output      = example["output"]

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
        return {"text": prompt + output}
    else:
        return {"text": prompt}


def main():
    print("=" * 60)
    print("Assignment 3 - Alpaca Data Preparation")
    print("=" * 60)

    # ── Step 1: Download dataset ───────────────────────────────────────────────
    print(f"\n[1/4] Downloading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"      Raw dataset size: {len(dataset)} examples")

    # ── Step 2: Format and clean ───────────────────────────────────────────────
    print("\n[2/4] Formatting and cleaning examples...")
    formatted = []
    skipped   = 0

    for example in dataset:
        result = format_example(example)
        if result is None:
            skipped += 1
        else:
            formatted.append(result)

    print(f"      Kept:    {len(formatted)} examples")
    print(f"      Skipped: {skipped} malformed examples")

    # ── Step 3: Shuffle and split ──────────────────────────────────────────────
    print(f"\n[3/4] Shuffling and splitting (eval={EVAL_SPLIT_SIZE*100:.0f}%)...")
    random.seed(RANDOM_SEED)
    random.shuffle(formatted)

    split_idx  = int(len(formatted) * (1 - EVAL_SPLIT_SIZE))
    train_data = formatted[:split_idx]
    eval_data  = formatted[split_idx:]

    print(f"      Train set: {len(train_data)} examples")
    print(f"      Eval set:  {len(eval_data)} examples")

    # ── Step 4: Save to disk ───────────────────────────────────────────────────
    print(f"\n[4/4] Saving to {OUTPUT_DIR}...")

    with open(TRAIN_FILE, "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"      Saved train → {TRAIN_FILE}")

    with open(EVAL_FILE, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"      Saved eval  → {EVAL_FILE}")

    # ── Preview a sample ───────────────────────────────────────────────────────
    print("\n── Sample training example ─────────────────────────────────")
    sample = train_data[0]
    print(f"  Instruction: {sample['instruction'][:80]}...")
    print(f"  Input:       {sample['input'][:80] if sample['input'] else '(none)'}")
    print(f"  Output:      {sample['output'][:80]}...")

    print("\n── Sample formatted text ───────────────────────────────────")
    formatted_sample = apply_chat_template(sample, include_output=True)
    print(formatted_sample["text"][:300])

    print("\n✅ Alpaca data preparation complete!")
    print(f"   Train: {TRAIN_FILE}")
    print(f"   Eval:  {EVAL_FILE}")


if __name__ == "__main__":
    main()