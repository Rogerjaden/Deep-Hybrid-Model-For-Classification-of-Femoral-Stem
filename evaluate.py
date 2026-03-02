import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from models.msftnet import MSFTNet
from utils.dataset import HipDataset
from utils.transforms import get_valid_transforms
import os

OUTPUT_DIR = "results/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "msftnet_model.pth"
DATASET_PATH = "dataset"
CLASS_NAMES = ["anatomical", "cemented", "uncemented"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD DATA
# -----------------------------
dataset = HipDataset(DATASET_PATH, transform=get_valid_transforms())
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = MSFTNet(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []
all_probs = []

# -----------------------------
# EVALUATION LOOP
# -----------------------------
with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# -----------------------------
# FINAL ACCURACY
# -----------------------------
accuracy = np.mean(all_preds == all_labels)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# -----------------------------
# ROC CURVE (Multi-class)
# -----------------------------
all_labels_bin = label_binarize(all_labels, classes=[0,1,2])

plt.figure()

for i in range(len(CLASS_NAMES)):
    fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

print("\nEvaluation complete.")
print("Saved: evaluation.png & roc_curve.png")


# -----------------------------
# COMPUTE FINAL METRICS
# -----------------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted")
recall = recall_score(all_labels, all_preds, average="weighted")
f1_macro = f1_score(all_labels, all_preds, average="macro")
f1_weighted = f1_score(all_labels, all_preds, average="weighted")

# Multi-class ROC AUC (OVR)
roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

print("\nDetailed Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")
print(f"ROC AUC (OVR): {roc_auc:.4f}")

# -----------------------------
# SAVE METRICS TO JSON
# -----------------------------
results = {
    "test_accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "roc_auc": float(roc_auc),
    "f1_macro": float(f1_macro),
    "f1_weighted": float(f1_weighted)
}

os.makedirs("results", exist_ok=True)

with open("results/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nMetrics saved to results/metrics.json")