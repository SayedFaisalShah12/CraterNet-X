# CraterNet-X: Detailed Methodology

This document outlines the technical approach, data pipeline, and decision-making process behind the CraterNet-X system.

## 1. Data Pipeline & Verification

### 1.1 Source Data
The primary dataset used is the **"Martian & Lunar Crater Detection Dataset"** from Kaggle. For the current phase of the research, the system is exclusively trained on **Lunar (Moon)** imagery to ensure high domain specificity.

### 1.2 Verification Process (`scripts/verify_dataset.py`)
To maintain research integrity, a custom verification script was developed to:
- Check for 1:1 parity between images and YOLO-format labels.
- Validate normalization coordinates (standardizing bounding boxes between 0 and 1).
- Screen for corrupted or unreadable images.

## 2. Stage 1: Object Detection (YOLOv8)

### 2.1 Model Choice
We chose **YOLOv8n (Nano)** for its exceptional speed and competitive mAP (mean Average Precision). In a planetary science context, the ability to process large-scale lunar maps rapidly is a priority.

### 2.2 Training Strategy
- **Optimizer**: AdamW for robust convergence.
- **Input Size**: 640x640 pixels.
- **Epochs**: 25-50 (depending on convergence).
- **GPU Usage**: Cloud-based training on NVIDIA T4 (Colab) to handle heavy computations.

## 3. Stage 2: Size Classification (ResNet-50)

### 3.1 Automated Cropping (`scripts/crop_craters.py`)
Detected objects from Stage 1 are passed through a cropping pipeline. A **10% padding** is added to each bounding box to provide the classifier with a small amount of context around the crater rim.

### 3.2 Categorization Thresholds
Crater sizes are categorized based on their maximum normalized dimension ($S = \max(w, h)$):
- **Small**: $S < 0.05$
- **Medium**: $0.05 \leq S < 0.10$
- **Large**: $S \geq 0.10$

These thresholds were empirically derived using `scripts/analyze_sizes.py`, which analyzed the statistical distribution (percentiles) of the training set.

### 3.3 Classification Model
A **ResNet-50** architecture was selected for the classification stage. 
- **Transfer Learning**: The model utilizes ImageNet pre-trained weights for feature extraction.
- **Fine-tuning**: The final fully-connected (FC) layer was replaced with a 3-unit output layer (Softmax) representing the three size categories.

## 4. Evaluation Metrics
The system is evaluated using a dual-metric approach:
1.  **Detection Performance**: mAP@.5 (Intersection over Union) and F1-score.
2.  **Classification Performance**: Accuracy and Confusion Matrix across the three size categories.

---
*For implementation details, see `inference/predict.py`.*
