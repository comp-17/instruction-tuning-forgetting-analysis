"""
Assignment 3 - Compute Metrics Script
Computes all automatic metrics for the three-checkpoint comparison table.

Metrics computed:
- ROUGE-1, ROUGE-2, ROUGE-L (for Alpaca evaluation)
- BERTScore (for Alpaca evaluation)
- JSON validity rate (for JSON evaluation)
- Schema compliance rate (for JSON evaluation)
- Exact match accuracy (for JSON evaluation)
- Field-level F1 (for JSON extraction tasks)
- Common error taxonomy (for JSON evaluation)
- Forgetting analysis (Checkpoint 1 vs Checkpoint 2 on Alpaca)

Usage:
    python compute_metrics.py
"""

import os
import json
import yaml
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from collections import defaultdict

# ── Load Config ────────────────────────────────────────────────────────────────
CONFIG_PATH = "/work/fpb170/assignment3/config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PATHS       = config["paths"]
RESULTS_DIR = PATHS["results_dir"]
BASE_DIR    = PATHS["base_dir"]
# ───────────────────────────────────────────────────────────────────────────────


def load_results(checkpoint_name, dataset_name):
    """Load inference results for a checkpoint."""
    path = os.path.join(RESULTS_DIR, f"{checkpoint_name}_{dataset_name}.json")
    if not os.path.exists(path):
        print(f"❌ Results not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ── ROUGE Metrics ──────────────────────────────────────────────────────────────
def compute_rouge(results):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1_scores, r2_scores, rl_scores = [], [], []

    for ex in results:
        reference = ex.get("reference", "")
        response  = ex.get("response", "")

        if not reference or not response:
            continue

        scores = scorer.score(reference, response)
        r1_scores.append(scores["rouge1"].fmeasure)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": round(sum(r1_scores) / len(r1_scores), 4) if r1_scores else 0,
        "rouge2": round(sum(r2_scores) / len(r2_scores), 4) if r2_scores else 0,
        "rougeL": round(sum(rl_scores) / len(rl_scores), 4) if rl_scores else 0,
        "count":  len(r1_scores),
    }


# ── BERTScore ──────────────────────────────────────────────────────────────────
def compute_bertscore(results):
    """Compute BERTScore F1."""
    references = [ex.get("reference", "") for ex in results]
    responses  = [ex.get("response",  "") for ex in results]

    # Filter empty
    pairs = [(r, p) for r, p in zip(references, responses) if r and p]
    if not pairs:
        return {"bertscore_f1": 0, "count": 0}

    refs, preds = zip(*pairs)

    P, R, F1 = bert_score(
        list(preds),
        list(refs),
        lang="en",
        verbose=False,
    )

    return {
        "bertscore_f1": round(F1.mean().item(), 4),
        "count": len(pairs),
    }


# ── Output Length ──────────────────────────────────────────────────────────────
def compute_output_length(results):
    """Compute average output length in tokens (approximate)."""
    lengths = [
        len(ex.get("response", "").split())
        for ex in results
        if ex.get("response")
    ]
    return {
        "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "count": len(lengths),
    }


# ── Task Completion Rate ───────────────────────────────────────────────────────
def compute_task_completion(results):
    """Estimate task completion rate (response is non-empty and reasonable length)."""
    completed = sum(
        1 for ex in results
        if ex.get("response") and len(ex["response"].strip()) > 10
    )
    total = len(results)
    return {
        "task_completion_rate": round(completed / total, 4) if total else 0,
        "completed": completed,
        "total": total,
    }


# ── JSON Validity ──────────────────────────────────────────────────────────────
def compute_json_validity(results):
    """Compute JSON validity rate."""
    valid   = 0
    invalid = 0
    errors  = defaultdict(int)

    for ex in results:
        response = ex.get("response", "").strip()

        # Strip markdown if present
        if response.startswith("```"):
            lines    = response.split("\n")
            response = "\n".join(lines[1:-1])

        # Extract JSON using brace counting
        brace_count = 0
        start = None
        extracted = None
        for i, c in enumerate(response):
            if c == '{':
                if start is None:
                    start = i
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0 and start is not None:
                    try:
                        extracted = json.loads(response[start:i+1])
                        break
                    except:
                        pass
        if extracted is not None:
            valid += 1
            continue
        try:
            json.loads(response)
            valid += 1
        except json.JSONDecodeError as e:
            invalid += 1
            # Categorize error type
            error_msg = str(e).lower()
            if "expecting" in error_msg and "'" in error_msg:
                errors["missing_quotes"] += 1
            elif "extra data" in error_msg:
                errors["extra_data"] += 1
            elif "unterminated" in error_msg:
                errors["unterminated_string"] += 1
            elif "expecting value" in error_msg:
                errors["missing_value"] += 1
            elif "trailing" in error_msg:
                errors["trailing_comma"] += 1
            else:
                errors["other"] += 1

    total = valid + invalid
    return {
        "json_validity_rate": round(valid / total, 4) if total else 0,
        "valid":   valid,
        "invalid": invalid,
        "total":   total,
        "error_taxonomy": dict(errors),
    }


# ── Schema Compliance ──────────────────────────────────────────────────────────
def compute_schema_compliance(results):
    """
    Compute schema compliance rate.
    Checks if valid JSON has the expected top-level keys from reference.
    """
    compliant     = 0
    valid_json    = 0
    non_compliant = 0

    for ex in results:
        response  = ex.get("response", "").strip()
        reference = ex.get("reference", "").strip()

        # Strip markdown
        if response.startswith("```"):
            lines    = response.split("\n")
            response = "\n".join(lines[1:-1])

        try:
            pred_json = json.loads(response)
            valid_json += 1

            try:
                ref_json = json.loads(reference)
                # Check if top-level keys match
                if isinstance(pred_json, dict) and isinstance(ref_json, dict):
                    ref_keys  = set(ref_json.keys())
                    pred_keys = set(pred_json.keys())
                    if ref_keys.issubset(pred_keys) or pred_keys.issubset(ref_keys):
                        compliant += 1
                    else:
                        non_compliant += 1
                else:
                    compliant += 1  # Arrays or primitives - just being valid JSON counts
            except:
                compliant += 1  # Reference not JSON - just being valid counts

        except:
            non_compliant += 1

    total = len(results)
    return {
        "schema_compliance_rate": round(compliant / total, 4) if total else 0,
        "compliant":    compliant,
        "valid_json":   valid_json,
        "non_compliant": non_compliant,
        "total":        total,
    }


# ── Exact Match ────────────────────────────────────────────────────────────────
def compute_exact_match(results):
    """Compute exact match accuracy between response and reference."""
    matches = 0
    total   = 0

    for ex in results:
        response  = ex.get("response",  "").strip()
        reference = ex.get("reference", "").strip()

        # Strip markdown
        if response.startswith("```"):
            lines    = response.split("\n")
            response = "\n".join(lines[1:-1])

        # Try JSON exact match (key-order independent)
        try:
            pred_json = json.loads(response)
            ref_json  = json.loads(reference)
            if pred_json == ref_json:
                matches += 1
        except:
            # Fall back to string exact match
            if response == reference:
                matches += 1

        total += 1

    return {
        "exact_match": round(matches / total, 4) if total else 0,
        "matches": matches,
        "total":   total,
    }


# ── Field-Level F1 ─────────────────────────────────────────────────────────────
def compute_field_level_f1(results):
    """
    Compute field-level precision, recall, and F1 for JSON extraction tasks.
    """
    precisions, recalls, f1s = [], [], []

    for ex in results:
        task_type = ex.get("task_type", "")
        if "extraction" not in task_type.lower():
            continue

        response  = ex.get("response",  "").strip()
        reference = ex.get("reference", "").strip()

        # Strip markdown
        if response.startswith("```"):
            lines    = response.split("\n")
            response = "\n".join(lines[1:-1])

        try:
            pred_json = json.loads(response)
            ref_json  = json.loads(reference)

            if not isinstance(pred_json, dict) or not isinstance(ref_json, dict):
                continue

            pred_items = set(f"{k}:{v}" for k, v in pred_json.items())
            ref_items  = set(f"{k}:{v}" for k, v in ref_json.items())

            if not ref_items:
                continue

            tp        = len(pred_items & ref_items)
            precision = tp / len(pred_items) if pred_items else 0
            recall    = tp / len(ref_items)  if ref_items  else 0
            f1        = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        except:
            continue

    if not f1s:
        return {"field_f1": 0, "field_precision": 0, "field_recall": 0, "count": 0}

    return {
        "field_precision": round(sum(precisions) / len(precisions), 4),
        "field_recall":    round(sum(recalls)    / len(recalls),    4),
        "field_f1":        round(sum(f1s)        / len(f1s),        4),
        "count":           len(f1s),
    }


# ── Forgetting Analysis ────────────────────────────────────────────────────────
def compute_forgetting_analysis(ckpt1_alpaca_metrics, ckpt2_alpaca_metrics):
    """
    Compare Alpaca metrics at Checkpoint 1 vs Checkpoint 2.
    This is the central analysis of the assignment.
    """
    print(f"\n{'='*60}")
    print("FORGETTING ANALYSIS")
    print("Checkpoint 1 (Alpaca) vs Checkpoint 2 (Teacher JSON)")
    print(f"{'='*60}")

    analysis = {}

    metrics_to_compare = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]

    for metric in metrics_to_compare:
        ckpt1_val = ckpt1_alpaca_metrics.get(metric, 0)
        ckpt2_val = ckpt2_alpaca_metrics.get(metric, 0)
        change    = round(ckpt2_val - ckpt1_val, 4)
        pct_change = round((change / ckpt1_val * 100) if ckpt1_val else 0, 2)

        analysis[metric] = {
            "checkpoint1": ckpt1_val,
            "checkpoint2": ckpt2_val,
            "absolute_change": change,
            "percent_change": pct_change,
            "verdict": (
                "FORGETTING" if change < -0.02
                else "MAINTAINED" if abs(change) <= 0.02
                else "IMPROVED"
            ),
        }

        print(f"\n{metric.upper()}:")
        print(f"  Checkpoint 1: {ckpt1_val}")
        print(f"  Checkpoint 2: {ckpt2_val}")
        print(f"  Change:       {change:+.4f} ({pct_change:+.1f}%)")
        print(f"  Verdict:      {analysis[metric]['verdict']}")

    return analysis


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Assignment 3 - Compute Metrics")
    print("=" * 60)

    checkpoints  = ["checkpoint0", "checkpoint1", "checkpoint2", "combined"]
    all_metrics  = {}

    # ── Alpaca Metrics ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("ALPACA EVALUATION METRICS")
    print(f"{'─'*60}")

    for ckpt in checkpoints:
        results = load_results(ckpt, "alpaca")
        if results is None:
            continue

        print(f"\n[{ckpt}] Computing Alpaca metrics ({len(results)} examples)...")

        rouge      = compute_rouge(results)
        bert       = compute_bertscore(results)
        length     = compute_output_length(results)
        completion = compute_task_completion(results)

        alpaca_metrics = {
            **rouge,
            **bert,
            **length,
            **completion,
        }

        all_metrics[f"{ckpt}_alpaca"] = alpaca_metrics

        print(f"  ROUGE-1:           {rouge['rouge1']}")
        print(f"  ROUGE-2:           {rouge['rouge2']}")
        print(f"  ROUGE-L:           {rouge['rougeL']}")
        print(f"  BERTScore F1:      {bert['bertscore_f1']}")
        print(f"  Avg length:        {length['avg_length']} words")
        print(f"  Task completion:   {completion['task_completion_rate']}")

    # ── JSON Metrics ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("JSON EVALUATION METRICS")
    print(f"{'─'*60}")

    for ckpt in checkpoints:
        results = load_results(ckpt, "json")
        if results is None:
            continue

        print(f"\n[{ckpt}] Computing JSON metrics ({len(results)} examples)...")

        validity   = compute_json_validity(results)
        compliance = compute_schema_compliance(results)
        exact      = compute_exact_match(results)
        field_f1   = compute_field_level_f1(results)

        json_metrics = {
            **validity,
            **compliance,
            **exact,
            **field_f1,
        }

        all_metrics[f"{ckpt}_json"] = json_metrics

        print(f"  JSON validity:     {validity['json_validity_rate']}")
        print(f"  Schema compliance: {compliance['schema_compliance_rate']}")
        print(f"  Exact match:       {exact['exact_match']}")
        print(f"  Field F1:          {field_f1['field_f1']}")
        print(f"  Error taxonomy:    {validity['error_taxonomy']}")

    # ── Forgetting Analysis ────────────────────────────────────────────────────
    if "checkpoint1_alpaca" in all_metrics and "checkpoint2_alpaca" in all_metrics:
        forgetting = compute_forgetting_analysis(
            all_metrics["checkpoint1_alpaca"],
            all_metrics["checkpoint2_alpaca"],
        )
        all_metrics["forgetting_analysis"] = forgetting

    # ── Three-Checkpoint Comparison Table ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("THREE-CHECKPOINT COMPARISON TABLE (Section 4.1)")
    print(f"{'='*60}")
    print(f"\n{'Checkpoint':<15} {'ROUGE-L':<10} {'BERTScore':<12} {'JSON Valid':<12} {'Schema':<10} {'Exact Match'}")
    print(f"{'─'*70}")

    for ckpt in checkpoints:
        alpaca = all_metrics.get(f"{ckpt}_alpaca", {})
        json_m = all_metrics.get(f"{ckpt}_json",   {})
        print(
            f"{ckpt:<15} "
            f"{alpaca.get('rougeL', 'N/A'):<10} "
            f"{alpaca.get('bertscore_f1', 'N/A'):<12} "
            f"{json_m.get('json_validity_rate', 'N/A'):<12} "
            f"{json_m.get('schema_compliance_rate', 'N/A'):<10} "
            f"{json_m.get('exact_match', 'N/A')}"
        )

    # ── Save All Metrics ───────────────────────────────────────────────────────
    output_path = os.path.join(RESULTS_DIR, "all_metrics.json")
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n✅ All metrics saved to {output_path}")
    print(f"\nNext step: Write REPORT.md using these results!")


if __name__ == "__main__":
    main()
