"""
Generate figures from saved metrics to reproduce all charts in REPORT.md.
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(f"{RESULTS_DIR}/all_metrics.json") as f:
    metrics = json.load(f)

checkpoints = ["checkpoint0", "checkpoint1", "checkpoint2"]
labels = ["Checkpoint 0\n(Base)", "Checkpoint 1\n(Alpaca)", "Checkpoint 2\n(JSON)"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

# Figure 5: ROUGE-L
rouge_l = [metrics[f"{c}_alpaca"]["rougeL"] for c in checkpoints]
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, rouge_l, color=colors, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, rouge_l):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.ylabel("ROUGE-L Score", fontsize=12)
plt.title("ROUGE-L Across Three Checkpoints", fontsize=14, fontweight="bold")
plt.ylim(0, max(rouge_l) * 1.15)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/rouge_l_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURES_DIR}/rouge_l_comparison.png")

# Figure 6: BERTScore
bertscore = [metrics[f"{c}_alpaca"]["bertscore_f1"] for c in checkpoints]
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, bertscore, color=colors, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, bertscore):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.ylabel("BERTScore F1", fontsize=12)
plt.title("BERTScore F1 Across Three Checkpoints", fontsize=14, fontweight="bold")
plt.ylim(min(bertscore) * 0.99, max(bertscore) * 1.005)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/bertscore_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURES_DIR}/bertscore_comparison.png")

# Figure 7: JSON Validity
json_validity = [metrics[f"{c}_json"]["json_validity_rate"] for c in checkpoints]
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, json_validity, color=colors, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, json_validity):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.ylabel("JSON Validity Rate", fontsize=12)
plt.title("JSON Validity Rate Across Three Checkpoints", fontsize=14, fontweight="bold")
plt.ylim(0, 1.15)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/json_validity_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURES_DIR}/json_validity_comparison.png")

# Figure 8: Forgetting Analysis
fa = metrics["forgetting_analysis"]
metric_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"]
c1_vals = [fa["rouge1"]["checkpoint1"], fa["rouge2"]["checkpoint1"],
           fa["rougeL"]["checkpoint1"], fa["bertscore_f1"]["checkpoint1"]]
c2_vals = [fa["rouge1"]["checkpoint2"], fa["rouge2"]["checkpoint2"],
           fa["rougeL"]["checkpoint2"], fa["bertscore_f1"]["checkpoint2"]]

x = range(len(metric_names))
width = 0.35
plt.figure(figsize=(10, 6))
plt.bar([i - width/2 for i in x], c1_vals, width,
        label="Checkpoint 1 (Alpaca)", color="#DD8452", edgecolor="black", linewidth=0.8)
plt.bar([i + width/2 for i in x], c2_vals, width,
        label="Checkpoint 2 (JSON)", color="#55A868", edgecolor="black", linewidth=0.8)
plt.ylabel("Score", fontsize=12)
plt.title("Forgetting Analysis: Checkpoint 1 vs Checkpoint 2\n(No Catastrophic Forgetting Observed)", fontsize=13, fontweight="bold")
plt.xticks(x, metric_names, fontsize=11)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/forgetting_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURES_DIR}/forgetting_analysis.png")

print("\n✅ All figures generated successfully!")
