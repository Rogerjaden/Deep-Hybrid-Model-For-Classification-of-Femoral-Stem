# 🦴 MSFT-Net: Deep Hybrid Model for Automatic Femoral Stem Classification

---

## 🎯 What is this project about?

**MSFT-Net** is an advanced Artificial Intelligence research project developed to solve a critical challenge in orthopedic surgery: the rapid and accurate identification of femoral stem implants from hip X-ray radiographs.

By integrating multi-scale convolutional features with dual attention mechanisms — **CBAM (Convolutional Block Attention Module)** for joint spatial-channel recalibration and **ECA (Efficient Channel Attention)** for lightweight cross-channel interaction — this system automates the classification of implants into three primary categories (Anatomical, Cemented, and Uncemented). This tool is designed to support clinical decision-making, preoperative planning for revision surgeries, and large-scale clinical data verification with **93%+ accuracy** and **Explainable AI (Grad-CAM)** visualizations.

---

## 📌 Project Overview

This project presents a medical AI system for automatically classifying femoral stem implant types from hip X-ray radiographs using a deep hybrid neural network architecture.

The proposed model, MSFT-Net (Multi-Scale Feature Transformer Network), combines:

- Convolutional Neural Networks (CNN)
- **CBAM** — Convolutional Block Attention Module (baseline attention, channel + spatial)
- **ECA** — Efficient Channel Attention (optimised final attention, 1D cross-channel)
- Transformer Encoder Modules

to accurately classify implant types into:

- 🟢 Anatomical
- 🔵 Cemented
- 🟡 Uncemented

---

## 🎯 Problem Statement

In hip arthroplasty (hip replacement surgery), identifying the correct femoral stem implant type is critical for:

- Revision surgeries
- Pre-operative planning
- Implant compatibility
- Clinical documentation verification

Manual identification from radiographs can be:

- Time-consuming
- Error-prone
- Dependent on expert knowledge

This project automates implant classification using deep learning.

---

---

## � Visual Results & Analytics

### 🔍 Grad-CAM: Input vs. Explanation
Grad-CAM (Gradient-weighted Class Activation Mapping) helps visualize which parts of the X-ray the model prioritized for its classification.

| Sample Input (Raw X-Ray) | Model Explanation (Grad-CAM) |
|:---:|:---:|
| ![Input](test_images/loose%20(39).png) | ![Grad-CAM](results/gradcam/loose%20(39).png) |
| *Example Input: Loose Stem* | *Heatmap highlighting the loose femoral stem region* |

---

### 📈 Model Metrics Plots
The following plots illustrate the performance and stability of MSFT-Net across all classes.

| F1-Score Comparison | Key Model Metrics |
|:---:|:---:|
| ![F1 Comparison](results/plots/f1_score_comparison.png) | ![Key Metrics](results/plots/key_model_metrics.png) |

---

---

## 🚦 Project Status & Highlights
- **Performance:** Achieved **93.37% test accuracy** across three classes.
- **Explainability:** Integrated **Grad-CAM** heatmaps to provide visual evidence for clinical trust.
- **Attention Baseline:** Implemented **CBAM (Convolutional Block Attention Module)** with sequential channel + spatial gates as the initial attention design.
- **Attention Upgrade:** Migrated to **ECA (Efficient Channel Attention)** — a parameter-efficient 1D convolution approach — for superior speed and accuracy in medical imaging.
- **Ready-to-Run:** Includes a **Mock Simulation** environment to test full pipeline functionality instantly.

---

## 🧠 Proposed Architecture: MSFT-Net

The architecture integrates:

- Multi-Scale Feature Extraction
- Pretrained ResNet-50 backbone (ImageNet weights via `timm`)
- **CBAM (Convolutional Block Attention Module)** *(baseline, explored during development)*
  - Sequential channel attention → spatial attention
  - Channel attention via shared MLP on avg-pool & max-pool descriptors
  - Spatial attention via 7×7 convolution on aggregated feature maps
- **ECA (Efficient Channel Attention)** *(selected for final model)*
  - Avoids dimensionality reduction; uses 1D convolution for cross-channel interaction
  - Kernel size adaptively determined by channel depth (log-based formula)
  - Extremely lightweight: ~0 extra parameters relative to CBAM
- Transformer Encoder Module
  - Captures long-range global contextual relationships across spatial tokens
- Fully Connected Classifier

### 🗺 Model Architecture Flow
```mermaid
flowchart TD

%% =========================
%% Legend
%% =========================
subgraph "Legend"
direction TB
L1["Blue: Entry scripts (pipelines)"]:::script
L2["Orange: Shared utilities (data/engine/metrics)"]:::util
L3["Green: Model code (MSFT-Net components)"]:::model
L4[("Gray cylinder: Artifacts in results/")]:::artifact
L5["Solid arrow: data/artifact flow"]:::note
L6["Dashed arrow: code dependency"]:::note
end

%% =========================
%% Swimlane 1: Inputs
%% =========================
subgraph "Inputs"
direction TB
DATASET["dataset/ (X-ray images)\n- anatomical/\n- cemented/\n- uncemented/"]:::data
TESTIMAGES["test_images/ (new X-ray inputs)"]:::data
end

%% =========================
%% Swimlane 2: Shared Libraries
%% =========================
subgraph "Core Libraries (shared)"
direction TB

subgraph "utils/ (PyTorch + Albumentations + scikit-learn)"
direction TB
UT_INIT["utils/__init__.py\n(package exports)"]:::util
UT_TRANS["utils/transforms.py\naugmentation + preprocessing\n(Albumentations/torchvision)"]:::util
UT_DATA["utils/dataset.py\nDataset/DataLoader wiring\n(PyTorch)"]:::util
UT_ENGINE["utils/engine.py\ntrain/val loops + device mgmt\n(PyTorch)"]:::util
UT_METRICS["utils/metrics.py\nF1/ROC-AUC/confusion matrix\n(scikit-learn)"]:::util
end

subgraph "models/ (PyTorch)"
direction TB
M_INIT["models/__init__.py\n(package exports)"]:::model
MSFT["models/msftnet.py\nMSFT-Net (CNN + ECA + Transformer + Classifier)"]:::model
ECA["models/eca.py\nECA attention module"]:::model
GCAM["gradcam.py\nExplainability Engine\n(Grad-CAM heatmaps)"]:::model
end
end

%% =========================
%% Swimlane 3: Runners (entry points)
%% =========================
subgraph "Runners (entry points)"
direction TB
TRAIN["main.py\nOffline training pipeline\n(3-class classification)"]:::script
EVAL["evaluate.py\nOffline evaluation pipeline\n(metrics + eval outputs)"]:::script
PRED["predict.py\nInference/prediction pipeline\n(Inference + Grad-CAM)"]:::script
REPORT["final_metrics_plot.py\nReporting/visualization pipeline\n(Matplotlib/Seaborn)"]:::script
end

%% =========================
%% Swimlane 4: Artifacts (results/)
%% =========================
subgraph "Artifacts (results/)"
direction TB
RESULTS[("results/ (artifact store)")]:::artifact
METJSON[("results/metrics.json\ncanonical metrics artifact")]:::artifact
EVALOUT[("results/evaluation/\nconfusion matrix + ROC + per-class outputs")]:::artifact
PLOTS[("results/plots/\ntraining curves + metric comparisons")]:::artifact
GCAMOUT[("results/gradcam/\nGrad-CAM overlay images")]:::artifact
end

%% =========================
%% Data flow: Inputs -> utils -> runners -> artifacts
%% =========================
DATASET -->|"images"| UT_TRANS -->|"tensors"| UT_DATA
TESTIMAGES -->|"images"| UT_TRANS

UT_DATA -->|"batches"| TRAIN
UT_DATA -->|"test-batches"| EVAL

UT_TRANS -->|"preprocessed-image"| PRED

%% =========================
%% Code dependencies (dashed)
%% =========================
TRAIN -.->|"uses"| UT_ENGINE
TRAIN -.->|"uses"| UT_METRICS
TRAIN -.->|"uses"| MSFT

EVAL -.->|"uses"| UT_DATA
EVAL -.->|"uses"| UT_METRICS
EVAL -.->|"loads"| MSFT

PRED -.->|"loads"| MSFT
PRED -.->|"uses"| GCAM

REPORT -.->|"reads"| METJSON

MSFT -.->|"uses"| ECA
UT_INIT -.->|"exports"| UT_TRANS
UT_INIT -.->|"exports"| UT_DATA
UT_INIT -.->|"exports"| UT_ENGINE
UT_INIT -.->|"exports"| UT_METRICS
M_INIT -.->|"exports"| MSFT
M_INIT -.->|"exports"| ECA

%% =========================
%% Artifact flow
%% =========================
TRAIN -->|"trained-model (saved weights)"| RESULTS
EVAL -->|"writes"| METJSON
EVAL -->|"writes"| EVALOUT
METJSON -->|"input"| REPORT
REPORT -->|"writes"| PLOTS
PRED -->|"writes heatmaps"| GCAMOUT

RESULTS -->|"contains"| METJSON
RESULTS -->|"contains"| EVALOUT
RESULTS -->|"contains"| PLOTS
RESULTS -->|"contains"| GCAMOUT

%% =========================
%% Model Internal Architecture (MSFT-Net)
%% =========================
subgraph "MSFT-Net internal architecture (from models/msftnet.py)"
direction TB
XIN["Input: X-ray tensor\n(after transforms)"]:::model
BACKB["Backbone: Pretrained ResNet\n(torchvision/timm)"]:::model
MSCALE["Multi-scale feature maps\n(C2/C3/C4/C5)"]:::model
ATTN["ECA attention\n(models/eca.py)"]:::model
TRANS["Transformer encoder\n(global context modeling)"]:::model
HEAD["Classifier head\n(FC layers)"]:::model
OUT["Output: Softmax logits\n3 classes:\nAnatomical / Cemented / Uncemented"]:::model

XIN --> BACKB --> MSCALE --> ATTN --> TRANS --> HEAD --> OUT
end

%% Tie shared model node to internal view
MSFT -.->|"implements"| XIN

%% =========================
%% Click events (from component mapping)
%% =========================
click TRAIN "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/main.py"
click EVAL "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/evaluate.py"
click PRED "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/predict.py"
click GCAM "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/gradcam.py"
click REPORT "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/final_metrics_plot.py"

click MSFT "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/models/msftnet.py"
click ECA "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/models/eca.py"
click M_INIT "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/models/__init__.py"

click UT_DATA "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/utils/dataset.py"
click UT_TRANS "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/utils/transforms.py"
click UT_ENGINE "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/utils/engine.py"
click UT_METRICS "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/utils/metrics.py"
click UT_INIT "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/utils/__init__.py"

click RESULTS "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/tree/main/results/"
click METJSON "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/blob/main/results/metrics.json"
click EVALOUT "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/tree/main/results/evaluation/"
click GCAMOUT "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/tree/main/results/gradcam/"
click PLOTS "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/tree/main/results/plots/"

click TESTIMAGES "https://github.com/rogerjaden/deep-hybrid-model-for-classification-of-femoral-stem/tree/main/test_images/"

%% =========================
%% Styles
%% =========================
classDef script fill:#1f77b4,stroke:#0b3d66,color:#ffffff,stroke-width:1px
classDef util fill:#ff7f0e,stroke:#7a3a00,color:#ffffff,stroke-width:1px
classDef model fill:#2ca02c,stroke:#145214,color:#ffffff,stroke-width:1px
classDef artifact fill:#9aa0a6,stroke:#4b4f54,color:#111111,stroke-width:1px
classDef data fill:#9467bd,stroke:#3f2a63,color:#ffffff,stroke-width:1px
classDef note fill:#f2f2f2,stroke:#cccccc,color:#111111,stroke-width:1px
```

---

---

## � Attention Mechanisms: CBAM vs ECA

A key research contribution of MSFT-Net is the comparative study and integration of two powerful attention strategies within the hybrid CNN-Transformer pipeline.

### 🔷 CBAM — Convolutional Block Attention Module

CBAM applies attention **sequentially** in two complementary dimensions:

1. **Channel Attention** (`ChannelAttention` in `models/cbam.py`)
   - Computes a global descriptor using **average pooling**.
   - Passes it through a shared **MLP** (FC → ReLU → FC) with reduction ratio `r=16`.
   - Outputs per-channel scaling weights via **Sigmoid**.
   - Recalibrates the feature map to emphasise informative channels.

2. **Spatial Attention** (`SpatialAttention` in `models/cbam.py`)
   - Aggregates channel information using both **average** and **max** pooling across channels.
   - Concatenates the two descriptors and applies a **7×7 convolution** followed by **Sigmoid**.
   - Produces a spatial mask highlighting the most discriminative regions of the X-ray.

```
Input → [Channel Attention] → weighted feature map → [Spatial Attention] → refined output
```

> **Strength:** Dual-axis attention provides rich, fine-grained spatial awareness — useful for localising implant boundaries in radiographs.

---

### 🔶 ECA — Efficient Channel Attention

ECA offers a **parameter-efficient** alternative to CBAM's MLP-based channel attention (`ECAModule` in `models/eca.py`):

1. **Global Average Pooling** reduces the spatial dimensions to a `[B, C, 1, 1]` descriptor.
2. The descriptor is reshaped to `[B, 1, C]` for **1D Convolution** across channels.
3. Kernel size `k` is computed adaptively:
   ```
   t = |( log₂(C) + b ) / γ|    (γ=2, b=1)
   k = t  (if odd)  else  t + 1
   ```
4. A **Sigmoid** gate generates the final channel weights.
5. The input feature map is element-wise multiplied by the attention weights.

```
Input → [AvgPool] → [1D Conv (adaptive k)] → [Sigmoid] → channel weights → weighted output
```

> **Strength:** No dimensionality reduction means no information bottleneck. Extremely fast, near-zero parameter overhead, and particularly effective on deeper feature maps (e.g., ResNet-50's 2048-channel C5 output).

---

### ⚖️ CBAM vs ECA — Head-to-Head Comparison

| Property | CBAM | ECA |
|---|---|---|
| **Attention Type** | Channel + Spatial (sequential) | Channel only |
| **Channel Mechanism** | MLP (FC layers, reduction ratio 16) | 1D Convolution (adaptive kernel) |
| **Spatial Mechanism** | 7×7 Conv on pooled maps | ❌ Not applicable |
| **Parameter Cost** | Higher (MLP weights + 7×7 conv) | Near-zero (single 1D conv) |
| **Dimensionality Reduction** | Yes (bottleneck) | No |
| **Inference Speed** | Moderate | Fast |
| **Kernel Adaptivity** | Fixed ratio | Adaptive per channel depth |
| **Role in MSFT-Net** | Baseline (explored) | Final model (selected) |

> **Why ECA was selected:** At 2048 channels (ResNet-50 C5 output), CBAM's MLP creates a large intermediate bottleneck that risks information loss. ECA's 1D convolution captures local cross-channel dependencies without this bottleneck, resulting in better accuracy and significantly lower compute overhead at scale.

---

## �📊 Dataset

- 2,744 hip X-ray images
- 3 implant classes:
  - Anatomical
  - Cemented
  - Uncemented

Images are organized as:

```text
dataset/
    anatomical/
    cemented/
    uncemented/
```

---

## 🏆 Model Performance

### Final Evaluation Results:

- ✅ Test Accuracy: 93.37%
- ✅ Macro F1-Score: 0.93
- ✅ Weighted F1-Score: 0.93
- ✅ ROC-AUC (OVR): ~0.93+

### Class-wise Performance:

| Class       | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Anatomical | 0.90      | 0.96   | 0.93     |
| Cemented   | 0.98      | 0.91   | 0.94     |
| Uncemented | 0.89      | 0.92   | 0.90     |

---

## 📈 Evaluation Outputs

The system automatically generates:

- Confusion Matrix
- ROC Curve
- F1 Score Comparison
- Key Metrics Visualization
- Training vs Validation Loss Graph
- Grad-CAM Visual Explanations
- **Persistent Text Logs** (Saved in `Logs/`)

All results are saved inside:

```text
results/
    evaluation/
    plots/
    gradcam/
Logs/
```

---

## 🔬 Explainability (Grad-CAM)

To improve interpretability in medical settings, Grad-CAM is implemented to:

- Highlight regions influencing predictions
- Verify model focuses on implant structure
- Improve trust in AI decisions

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

---

## 🚀 How to Run

### 1. Initial Setup
Run the automated installer to set up all dependencies:
```bash
./install_dependencies.bat
```

### 2. Mock Test (Optional)
If you don't have the dataset yet, run this to generate dummy data and weights for a dry run:
```bash
python setup_mock.py
```

### 3. Training
```bash
python main.py
```

### 4. Evaluation
```bash
python evaluate.py
```

### 5. Predict & Explain
Run prediction and generate Grad-CAM heatmaps on test images:
```bash
python predict.py
```

---

## 📁 Project Structure

```text
FinalYearProject/
│
├── dataset/             # Hip X-ray dataset (Anatomical, Cemented, Uncemented)
├── models/
│   ├── msftnet.py       # Main Hybrid Architecture
│   ├── eca.py           # Efficient Channel Attention Module
│   ├── cbam.py          # (Legacy) Convolutional Block Attention
│
├── utils/               # Dataset loaders, transforms, and training engine
├── results/             # Plots and Grad-CAM visualizations
├── Logs/                # Auto-generated CSV/Text logs for all runs
│
├── main.py              # Training Entry Point
├── evaluate.py          # Metric & Visualization Generator
├── predict.py           # Single-image inference + Explainability
├── requirements.txt     # Library dependencies
└── install_dependencies.bat # Windows Automated Setup Script
```

---

## 🔎 Future Improvements

- Formal ablation study: CBAM vs ECA vs SE-Net vs Coordinate Attention (quantitative comparison on test set)
- Re-enable CBAM as a switchable attention mode for research experiments
- Deploy as web application with live prediction and Grad-CAM overlay
- Integrate into PACS / DICOM viewing system
- Expand dataset with multi-hospital radiograph sources
- Perform cross-hospital and cross-scanner validation

---

## 🎓 Academic Context

This project was developed as a Final Year Research Project focusing on:

- Deep Learning in Medical Imaging
- Comparative Attention Mechanism Design (CBAM vs ECA)
- Hybrid CNN-Transformer Architectures
- Explainable AI in Healthcare (Grad-CAM)
- Efficient Model Design for Clinical Deployment

---

## ⚠ Disclaimer

This model is intended for research and academic purposes only and should not replace clinical judgment.
