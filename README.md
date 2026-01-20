# CraterNet-X: Automated Lunar Crater Analysis

CraterNet-X is a research-grade, two-stage deep learning pipeline designed for the automated detection and classification of craters on the Moon's surface using satellite imagery.

## 🚀 System Architecture

The pipeline consists of two distinct stages:

1.  **Stage 1: Detection (YOLOv8)**
    - Utilizes YOLOv8 (Nano) for high-speed, accurate detection of crater locations.
    - Trained on the "Martian & Lunar Crater Detection Dataset" (Moon subset).
    - Outputs bounding boxes for each detected crater.

2.  **Stage 2: Classification (ResNet-50)**
    - Automatically crops detected craters from the source imagery.
    - Uses a ResNet-50 CNN to categorize each crater into one of three size classes:
        - **Small** (Normalized size < 0.05)
        - **Medium** (0.05 <= size < 0.10)
        - **Large** (size >= 0.10)

## 📁 Project Structure

```text
CraterNet-X/
├── app/                    # Streamlit Demo Application
├── data/                   # Dataset storage (Moon & Classification crops)
├── evaluation/             # Evaluation scripts and visualization results
├── inference/              # Core inference pipeline (CraterNetX class)
├── models/                 # Trained model weights (.pt)
├── scripts/                # Utility scripts (verification, cropping, analysis)
├── training/               # Training scripts for YOLOv8 and ResNet-50
├── requirments.txt         # Project dependencies
└── README.md               # Project documentation
```

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- PyTorch (with CUDA support recommended)
- `pip install -r requirments.txt`

### 2. Dataset Verification
Ensure your dataset is correctly formatted and clean:
```bash
python scripts/verify_dataset.py
```

### 3. Training
**Stage 1 (Detection):**
```bash
python training/train_yolov8.py
```

**Prepare Classification Data:**
```bash
python scripts/crop_craters.py
```

**Stage 2 (Classification):**
```bash
python training/train_classifier.py
```

### 4. Running the Demo
Launch the interactive Streamlit dashboard:
```bash
streamlit run app/main.py
```

## 📊 Methodology & Research
This project follows a modular research approach. Each stage is independently verifiable. Classification thresholds were determined by analyzing the statistical distribution of the crater sizes in the training set (found in `scripts/analyze_sizes.py`).

## 📜 License
This project is licensed under the MIT License.
```
