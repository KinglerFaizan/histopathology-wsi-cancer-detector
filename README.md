# 🔬 Histopathology WSI Cancer Detector

> **Deep learning pipeline for tumor detection in Whole-Slide Images (WSI)** using EfficientNet/ResNet backbones, patch-level classification with slide-level aggregation, stain normalization, Grad-CAM explainability, and hard-negative mining — evaluated on the **PatchCamelyon (PCam) dataset**.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-PatchCamelyon-green)](https://github.com/basveeling/pcam)

---

## 📌 Problem Statement

Whole-Slide Images (WSIs) in digital pathology are extremely high-resolution (up to 50,000×50,000 pixels). Manual inspection by pathologists is time-consuming, subjective, and not scalable. This project builds a complete automated pipeline to:

- **Tile** gigapixel WSIs into manageable 256×256 patches
- **Normalize** stain variations across slides from different labs
- **Classify** each patch as tumor or non-tumor
- **Aggregate** patch-level predictions to slide-level diagnosis
- **Explain** predictions using Grad-CAM heatmaps

---

## 🏗️ Project Structure

```
histopathology-wsi-cancer-detector/
├── src/
│   ├── dataset.py              # PatchCamelyon dataset loader & augmentation
│   ├── wsi_tiling.py           # WSI tiling engine (OpenSlide-based)
│   ├── stain_normalization.py  # Macenko & Reinhard stain normalization
│   ├── model.py                # EfficientNet-B4 / ResNet-50 classifier
│   ├── train.py                # Training loop with hard-negative mining
│   ├── evaluate.py             # AUC, F1, confusion matrix
│   ├── slide_inference.py      # Patch → slide-level aggregation
│   └── gradcam.py              # Grad-CAM heatmap visualization
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Inference_and_GradCAM.ipynb
├── configs/
│   └── config.yaml             # All hyperparameters in one place
├── data/
│   └── README.md               # Dataset download instructions
├── outputs/                    # Saved models, plots, heatmaps
├── requirements.txt
├── train.py                    # Entry point for training
└── README.md
```

---

## 🔬 Methodology

### 1. WSI Tiling
- Load 50k×50k gigapixel slides using **OpenSlide**
- Extract non-overlapping **256×256 patches** at 20× magnification
- Filter out background patches using **Otsu thresholding** on tissue content
- Store patches with coordinates for spatial reconstruction

### 2. Stain Normalization
- **Macenko method**: SVD-based stain matrix estimation, normalizes H&E stains to a target slide
- **Reinhard method**: Color space transfer using mean/std matching in LAB space
- Reduces inter-lab and inter-scanner staining variability

### 3. Data Augmentation (Heavy)
| Augmentation | Parameters |
|---|---|
| Random Horizontal/Vertical Flip | p=0.5 |
| Random Rotation | ±45° |
| Color Jitter | brightness=0.3, contrast=0.3, saturation=0.2 |
| Random Elastic Distortion | α=120, σ=8 |
| Gaussian Blur | kernel=3, p=0.3 |
| Random Grayscale | p=0.1 |
| Normalize | ImageNet mean/std |

### 4. Model Architecture
- **Backbone**: EfficientNet-B4 (primary) / ResNet-50 (baseline)
- **Head**: Global Average Pooling → Dropout(0.4) → FC(256) → ReLU → FC(1)
- **Loss**: Binary Cross-Entropy with class-weight balancing
- **Optimizer**: AdamW, lr=1e-4, weight_decay=1e-4
- **Scheduler**: CosineAnnealingLR with warm restarts

### 5. Hard-Negative Mining
- After epoch 3, collect all **false negatives** (missed tumors) and **false positives**
- Oversample hard examples at 3× rate in subsequent epochs
- Dramatically improves sensitivity on edge-case patches

### 6. Slide-Level Aggregation
- **Max pooling**: slide score = max patch probability
- **Mean pooling**: slide score = mean of top-K patch probabilities
- **MIL (Multiple Instance Learning)**: attention-weighted aggregation

### 7. Grad-CAM Explainability
- Hooks into the last convolutional block
- Generates class activation maps highlighting tumor-suspicious regions
- Overlaid on original H&E patches for pathologist review

---

## 📊 Results on PatchCamelyon (PCam)

| Model | AUC | F1 Score | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| ResNet-50 (baseline) | 0.68 | 0.81 | 0.82 | 0.74 | 0.89 |
| EfficientNet-B4 | **0.74** | **0.88** | **0.86** | **0.81** | **0.91** |
| EfficientNet-B4 + HNM | 0.74 | 0.88 | 0.87 | 0.84 | 0.90 |

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/KinglerFaizan/histopathology-wsi-cancer-detector.git
cd histopathology-wsi-cancer-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download PatchCamelyon dataset
```bash
# Instructions in data/README.md
# Dataset: https://github.com/basveeling/pcam
```

### 4. Train the model
```bash
python train.py --config configs/config.yaml --model efficientnet_b4
```

### 5. Evaluate
```bash
python src/evaluate.py --checkpoint outputs/best_model.pth
```

### 6. Run Grad-CAM on a patch
```bash
python src/gradcam.py --image path/to/patch.png --checkpoint outputs/best_model.pth
```

### 7. Run notebooks
```bash
jupyter notebook notebooks/
```

---

## 📦 Dataset

**PatchCamelyon (PCam)**
- 327,680 color images (96×96 px) extracted from histopathology scans
- Binary labels: tumor tissue (1) vs. normal tissue (0)
- Source: CAMELYON16 challenge, Radboud University Medical Center
- Download: [github.com/basveeling/pcam](https://github.com/basveeling/pcam)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch 2.0, torchvision |
| WSI Processing | OpenSlide, PIL, OpenCV |
| Models | EfficientNet-B4, ResNet-50 (timm) |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| Augmentation | Albumentations |
| Metrics | scikit-learn, torchmetrics |
| Visualization | Matplotlib, Seaborn |
| Config | PyYAML |
| Notebooks | Jupyter |

---

## 👤 Author

**Mohammed Faizan**
B.Tech, NIT Surat | Data Scientist
📧 faizanmohammed7833@gmail.com
🔗 [GitHub](https://github.com/KinglerFaizan)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
