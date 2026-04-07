"""
Assignment 3 - Judge Evaluation Script
Uses Llama 3.3 70B as judge to evaluate model outputs at each checkpoint.

Following assignment Section 4.2 (Alpaca Evaluation) and Section 4.3 (JSON Evaluation):
- Pairwise comparison between checkpoints
- Scores on 6 dimensions: instruction_following, correctness, clarity,
  completeness, structured_output_validity, hallucination_risk
- Randomizes response order to reduce position bias
- Comparisons: checkpoint0 vs 1, checkpoint1 vs 2, checkpoint0 vs 2

Usage:
    python judge_eval.py --comparison alpaca
    python judge_eval.py --comparison json
    python judge_eval.py --comparison all
"""

import os
import json
import yaml
import argparse
import random
from tqdm import tqdm
from openai import OpenAI

# ── Load Config ────────────────────────────────────────────────────────────────
CONFIG_PATH = "/work/fpb170/assignment3/config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PATHS      = config["paths"]
JUDGE_CFG  = config["judge"]
EVAL_CFG   = config["evaluation"]
BASE_DIR   = PATHS["base_dir"]
RESULTS_DIR = PATHS["results_dir"]
RANDOM_SEED = 42

# Initialize judge model client (Llama 3.3 70B)
client = OpenAI(
    api_key=os.environ.get("UTSA_API_KEY_70B"),
    base_url=os.environ.get("UTSA_BASE_URL_70B"),
)
JUDGE_MODEL = os.environ.get("UTSA_MODEL_70B")
# ───────────────────────────────────────────────────────────────────────────────


ALPACA_JUDGE_PROMPT = """You are an expert judge evaluating the quality of AI assistant responses.
You will be shown an instruction and two responses (Response A and Response B).
Your task is to evaluate both responses and determine which is better.

Instruction:
{instruction}

Input (if any):
{input}

Response A:
{response_a}

Response B:
{response_b}

Please evaluate both responses on these 6 dimensions (score 1-5 each):
1. instruction_following: How well does the response follow the instruction?
2. correctness: Is the response factually correct and accurate?
3. clarity: Is the response clear and well-written?
4. completeness: Does the response fully address the instruction?
5. structured_output_validity: Is any structured output (JSON, lists, etc.) valid and well-formed?
6. hallucination_risk: How likely is the response to contain fabricated information? (5=no hallucination, 1=high hallucination)

Return ONLY a valid JSON object in exactly this format:
{{
    "response_a_scores": {{
        "instruction_following": <1-5>,
        "correctness": <1-5>,
        "clarity": <1-5>,
        "completeness": <1-5>,
        "structured_output_validity": <1-5>,
        "hallucination_risk": <1-5>
    }},
    "response_b_scores": {{
        "instruction_following": <1-5>,
        "correctness": <1-5>,
        "clarity": <1-5>,
        "completeness": <1-5>,
        "structured_output_validity": <1-5>,
        "hallucination_risk": <1-5>
    }},
    "winner": "<A|B|tie>",
    "justification": "<brief explanation of why one response is better>"
}}"""


JSON_JUDGE_PROMPT = """You are an expert judge evaluating the quality of AI assistant responses to structured JSON tasks.
You will be shown an instruction and two responses (Response A and Response B).

Instruction:
{instruction}

Input (if any):
{input}

Response A:
{response_a}

Response B:
{response_b}

Please evaluate both responses on these 6 dimensions (score 1-5 each):
1. instruction_following: How well does the response follow the instruction?
2. correctness: Is the JSON content correct and accurate?
3. clarity: Is the response clear and well-structured?
4. completeness: Does the response include all required fields?
5. structured_output_validity: Is the JSON syntactically valid and schema-compliant? (5=perfect JSON, 1=invalid)
6. hallucination_risk: Does the response contain fabricated or incorrect values? (5=no hallucination, 1=high hallucination)

Return ONLY a valid JSON object in exactly this format:
{{
    "response_a_scores": {{
        "instruction_following": <1-5>,
        "correctness": <1-5>,
        "clarity": <1-5>,
        "completeness": <1-5>,
        "structured_output_validity": <1-5>,
        "hallucination_risk": <1-5>
    }},
    "response_b_scores": {{
        "instruction_following": <1-5>,
        "correctness": <1-5>,
        "clarity": <1-5>,
        "completeness": <1-5>,
        "structured_output_validity": <1-5>,
        "hallucination_risk": <1-5>
    }},
    "winner": "<A|B|tie>",
    "justification": "<brief explanation focusing on JSON quality>"
}}"""


def call_judge(prompt, max_retries=3):
    """Call Llama 3.3 70B judge and parse JSON response."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI judge. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])

            result = json.loads(raw)

            # Validate required fields
            assert "response_a_scores" in result
            assert "response_b_scores" in result
            assert "winner" in result
            assert "justification" in result
            assert result["winner"] in ["A", "B", "tie"]

            return result

        except Exception as e:
            print(f"\n  ⚠ Attempt {attempt+1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                return None


def load_results(checkpoint_name, dataset_name):
    """Load inference results for a checkpoint."""
    path = os.path.join(RESULTS_DIR, f"{checkpoint_name}_{dataset_name}.json")
    if not os.path.exists(path):
        print(f"❌ ERROR: Results not found at {path}")
        print(f"   Run inference.py --checkpoint {checkpoint_name} first!")
        return None
    with open(path) as f:
        return json.load(f)


def run_pairwise_comparison(
    checkpoint_a, checkpoint_b, dataset_name, judge_prompt_template
):
    """
    Run pairwise judge comparison between two checkpoints.
    Randomizes A/B order to reduce position bias.
    """
    print(f"\n[Judge] Comparing {checkpoint_a} vs {checkpoint_b} on {dataset_name}")

    # Load results
    results_a = load_results(checkpoint_a, dataset_name)
    results_b = load_results(checkpoint_b, dataset_name)

    if results_a is None or results_b is None:
        return None

    # Match examples by id
    results_b_dict = {r["id"]: r for r in results_b}

    random.seed(RANDOM_SEED)
    comparisons = []

    for ex_a in tqdm(results_a, desc=f"{checkpoint_a} vs {checkpoint_b}"):
        ex_id = ex_a["id"]
        if ex_id not in results_b_dict:
            continue

        ex_b = results_b_dict[ex_id]

        # Randomize order to reduce position bias
        swapped = random.random() > 0.5
        if swapped:
            resp_a = ex_b["response"]
            resp_b = ex_a["response"]
        else:
            resp_a = ex_a["response"]
            resp_b = ex_b["response"]

        # Build judge prompt
        prompt = judge_prompt_template.format(
            instruction=ex_a["instruction"],
            input=ex_a.get("input", ""),
            response_a=resp_a,
            response_b=resp_b,
        )

        judge_result = call_judge(prompt)

        if judge_result is None:
            print(f"\n  ⚠ Skipping example {ex_id} - judge failed")
            continue

        # If swapped, reverse winner
        winner = judge_result["winner"]
        if swapped:
            if winner == "A":
                winner = "B"
            elif winner == "B":
                winner = "A"

        # Map winner back to checkpoint names
        if winner == "A":
            winner_name = checkpoint_a
        elif winner == "B":
            winner_name = checkpoint_b
        else:
            winner_name = "tie"

        comparisons.append({
            "prompt_id":          f"{dataset_name}_{ex_id}",
            "checkpoint_a":       checkpoint_a,
            "checkpoint_b":       checkpoint_b,
            "instruction":        ex_a["instruction"],
            "input":              ex_a.get("input", ""),
            "task_type":          ex_a.get("task_type", "general"),
            "response_a":         ex_a["response"],
            "response_b":         ex_b["response"],
            "response_a_scores":  judge_result["response_a_scores"] if not swapped else judge_result["response_b_scores"],
            "response_b_scores":  judge_result["response_b_scores"] if not swapped else judge_result["response_a_scores"],
            "winner":             winner_name,
            "justification":      judge_result["justification"],
            "swapped":            swapped,
        })

    return comparisons


def compute_win_rates(comparisons, checkpoint_a, checkpoint_b):
    """Compute win rates from comparisons."""
    total = len(comparisons)
    if total == 0:
        return {}

    wins_a  = sum(1 for c in comparisons if c["winner"] == checkpoint_a)
    wins_b  = sum(1 for c in comparisons if c["winner"] == checkpoint_b)
    ties    = sum(1 for c in comparisons if c["winner"] == "tie")

    # Average scores per dimension
    dims = list(JUDGE_CFG["dimensions"])
    avg_scores_a = {}
    avg_scores_b = {}

    for dim in dims:
        scores_a = [c["response_a_scores"].get(dim, 0) for c in comparisons]
        scores_b = [c["response_b_scores"].get(dim, 0) for c in comparisons]
        avg_scores_a[dim] = round(sum(scores_a) / len(scores_a), 2)
        avg_scores_b[dim] = round(sum(scores_b) / len(scores_b), 2)

    return {
        "checkpoint_a":       checkpoint_a,
        "checkpoint_b":       checkpoint_b,
        "total_comparisons":  total,
        "wins_a":             wins_a,
        "wins_b":             wins_b,
        "ties":               ties,
        "win_rate_a":         round(wins_a / total, 3),
        "win_rate_b":         round(wins_b / total, 3),
        "tie_rate":           round(ties / total, 3),
        "avg_scores_a":       avg_scores_a,
        "avg_scores_b":       avg_scores_b,
    }


def save_judge_results(comparisons, summary, checkpoint_a, checkpoint_b, dataset_name):
    """Save judge results and summary."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save detailed comparisons
    comparison_file = os.path.join(
        RESULTS_DIR,
        f"judge_{dataset_name}_{checkpoint_a}_vs_{checkpoint_b}.json"
    )
    with open(comparison_file, "w") as f:
        json.dump(comparisons, f, indent=2)

    # Save summary
    summary_file = os.path.join(
        RESULTS_DIR,
        f"judge_summary_{dataset_name}_{checkpoint_a}_vs_{checkpoint_b}.json"
    )
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Save] Comparisons → {comparison_file}")
    print(f"[Save] Summary     → {summary_file}")

    return summary_file


def print_summary(summary, checkpoint_a, checkpoint_b):
    """Print win rate summary to terminal."""
    print(f"\n{'─'*50}")
    print(f"Results: {checkpoint_a} vs {checkpoint_b}")
    print(f"{'─'*50}")
    print(f"Total comparisons: {summary['total_comparisons']}")
    print(f"{checkpoint_a} wins:  {summary['wins_a']} ({summary['win_rate_a']*100:.1f}%)")
    print(f"{checkpoint_b} wins:  {summary['wins_b']} ({summary['win_rate_b']*100:.1f}%)")
    print(f"Ties:              {summary['ties']} ({summary['tie_rate']*100:.1f}%)")
    print(f"\nAverage scores ({checkpoint_a}):")
    for dim, score in summary["avg_scores_a"].items():
        print(f"  {dim}: {score}")
    print(f"\nAverage scores ({checkpoint_b}):")
    for dim, score in summary["avg_scores_b"].items():
        print(f"  {dim}: {score}")


def run_alpaca_evaluation():
    """Run judge evaluation on Alpaca outputs."""
    print(f"\n{'='*60}")
    print("Judge Evaluation: Alpaca (General Instruction Following)")
    print(f"{'='*60}")

    comparisons_list = JUDGE_CFG["comparisons"]
    all_summaries = []

    for ckpt_a, ckpt_b in comparisons_list:
        comparisons = run_pairwise_comparison(
            ckpt_a, ckpt_b, "alpaca", ALPACA_JUDGE_PROMPT
        )

        if comparisons is None or len(comparisons) == 0:
            print(f"⚠ No comparisons generated for {ckpt_a} vs {ckpt_b}")
            continue

        summary = compute_win_rates(comparisons, ckpt_a, ckpt_b)
        save_judge_results(comparisons, summary, ckpt_a, ckpt_b, "alpaca")
        print_summary(summary, ckpt_a, ckpt_b)
        all_summaries.append(summary)

    # Save combined summary
    combined_path = os.path.join(RESULTS_DIR, "judge_alpaca_all_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n✅ Alpaca judge evaluation complete!")
    print(f"   Combined summary → {combined_path}")


def run_json_evaluation():
    """Run judge evaluation on JSON outputs."""
    print(f"\n{'='*60}")
    print("Judge Evaluation: JSON (Structured Output)")
    print(f"{'='*60}")

    comparisons_list = JUDGE_CFG["comparisons"]
    all_summaries = []

    for ckpt_a, ckpt_b in comparisons_list:
        comparisons = run_pairwise_comparison(
            ckpt_a, ckpt_b, "json", JSON_JUDGE_PROMPT
        )

        if comparisons is None or len(comparisons) == 0:
            print(f"⚠ No comparisons generated for {ckpt_a} vs {ckpt_b}")
            continue

        summary = compute_win_rates(comparisons, ckpt_a, ckpt_b)
        save_judge_results(comparisons, summary, ckpt_a, ckpt_b, "json")
        print_summary(summary, ckpt_a, ckpt_b)
        all_summaries.append(summary)

    # Save combined summary
    combined_path = os.path.join(RESULTS_DIR, "judge_json_all_summaries.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n✅ JSON judge evaluation complete!")
    print(f"   Combined summary → {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run judge evaluation for Assignment 3"
    )
    parser.add_argument(
        "--comparison",
        type=str,
        required=True,
        choices=["alpaca", "json", "all"],
        help="Which evaluation to run"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Assignment 3 - Judge Evaluation")
    print(f"Judge model: {JUDGE_MODEL}")
    print("=" * 60)

    if args.comparison == "alpaca":
        run_alpaca_evaluation()
    elif args.comparison == "json":
        run_json_evaluation()
    elif args.comparison == "all":
        run_alpaca_evaluation()
        run_json_evaluation()

    print(f"\n{'='*60}")
    print("Judge Evaluation Complete!")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"Next step: Run compute_metrics.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
