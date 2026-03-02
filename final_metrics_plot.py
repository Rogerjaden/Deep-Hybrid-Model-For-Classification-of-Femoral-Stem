import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os

OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("results/metrics.json", "r") as f:
    metrics_data = json.load(f)

test_accuracy = metrics_data["test_accuracy"]
precision = metrics_data["precision"]
recall = metrics_data["recall"]
roc_auc = metrics_data["roc_auc"]

f1_macro = metrics_data["f1_macro"]
f1_weighted = metrics_data["f1_weighted"]
# -----------------------------
# F1 Score Comparison Plot
# -----------------------------

plt.figure(figsize=(6,5))
f1_values = [f1_macro, f1_weighted]
f1_labels = ["Macro F1", "Weighted F1"]

bars = plt.bar(f1_labels, f1_values)

for i, v in enumerate(f1_values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')

plt.ylim(0,1)
plt.title("F1 Score Comparison")
plt.ylabel("Score")
plt.savefig(os.path.join(OUTPUT_DIR, "f1_score_comparison.png"))
plt.close()

# -----------------------------
# Key Metrics Plot
# -----------------------------

metrics = [
    test_accuracy,
    f1_macro,
    f1_weighted,
    roc_auc,
    precision,
    recall
]

metric_names = [
    "Test Accuracy",
    "F1 Macro",
    "F1 Weighted",
    "ROC AUC",
    "Precision",
    "Recall"
]

plt.figure(figsize=(10,6))
bars = plt.bar(metric_names, metrics)

for i, v in enumerate(metrics):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', rotation=45)

plt.xticks(rotation=45)
plt.ylim(0,1)
plt.title("Key Model Performance Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "key_model_metrics.png"))
plt.close()

print("Saved: f1_score_comparison.png & key_model_metrics.png")