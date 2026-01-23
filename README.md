# 🌑 CraterNet-X: Automated Planetary Analysis Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-red)](https://github.com/ultralytics/ultralytics)
[![ResNet-50](https://img.shields.io/badge/Classification-ResNet--50-green)](https://pytorch.org/hub/pytorch_vision_resnet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CraterNet-X is a high-performance, two-stage deep learning pipeline designed for the autonomous detection and morphological classification of planetary craters using satellite imagery. This research-grade tool currently supports **Lunar** analysis, with infrastructure ready for **Martian** expansion.

---

## 🚀 Key Features

- **Precision Detection**: Leverages YOLOv8-Nano trained on normalized lunar datasets.
- **Size Classification**: Stage-2 ResNet-50 architecture categorizes craters into *Small*, *Medium*, and *Large* with high fidelity.
- **Interactive Dashboard**: Streamlit-powered interface for real-time inference and report generation.
- **Dataset Verification**: Built-in scripts to validate label integrity and class distribution.

---

## 🏗️ System Architecture

The pipeline follows a modular **Detection-then-Classification** strategy:

1.  **Stage 1 (Detection)**: Input imagery is scanned by YOLOv8. Bounding boxes are localized and extracted.
2.  **Stage 2 (Classification)**: Extracted crops are normalized and passed through a ResNet-50 CNN to determine the crater's size class based on morphology and relative area.

---

## 📁 Project Structure

```text
CraterNet-X/
├── app/                    # Interactive Streamlit Application
├── data/                   # Dataset management (Moon/Mars)
├── inference/              # Core pipeline (Predict & Visualize)
├── models/                 # Trained weights and model definitions
├── scripts/                # Data preparation & debugging utilities
├── training/               # Stage-1 and Stage-2 training scripts
├── requirements.txt         # Project dependencies
└── README.md               # Documentation
```

---

## 🛠️ Installation & Setup

### 1. Environment Preparation
```bash
git clone https://github.com/SayedFaisalShah12/CraterNet-X
cd CraterNet-X
pip install -r requirements.txt
```

### 2. Dataset Integration
Place your `kaggle.json` in the root directory. Use the provided Jupyter notebook `CraterNetX_Colab.ipynb` for cloud training or use local scripts:
```bash
python scripts/verify_dataset.py
```

### 3. Pipeline Execution
**Train Detection (Stage 1):**
```bash
python training/train_yolov8.py
```

**Generate Classification Crops:**
```bash
python scripts/crop_craters.py
```

**Train Classification (Stage 2):**
```bash
python training/train_classifier.py
```

---

## 📊 Evaluation & Results

The system integrates visualization tools in the `evaluation/` directory. Use the **CraterNet-X Interactive Dashboard** for a professional presentation of results:

```bash
streamlit run app/main.py
```

---

## 📜 License
Published under the **MIT License**. See `LICENSE` for more details.

## 🤝 Contributing
Contributions for Martian dataset integration and multi-spectral imagery support are welcome. 

Developed by **Sayed Faisal Shah**
