# Reproduction Guide: Training CraterNet-X in the Cloud

This guide provides step-by-step instructions to reproduce the trained models using Google Colab's GPU infrastructure.

## Step 1: Prepare GitHub Repository
Ensure your latest local scripts are pushed to GitHub:
```bash
git add .
git commit -m "Reproduction sync"
git push origin main
```

## Step 2: Google Colab Setup
1.  Open [Google Colab](https://colab.research.google.com/).
2.  Upload `CraterNetX_Colab.ipynb` from the project root.
3.  **Hardware Acceleration**: Go to `Runtime > Change runtime type` and select **T4 GPU**.

## Step 3: Kaggle Authentication
The automated dataset script requires your Kaggle API token:
1.  Go to [Kaggle Settings](https://www.kaggle.com/settings).
2.  Click **Create New API Token** to download `kaggle.json`.
3.  Upload this `kaggle.json` to the file explorer in the Colab sidebar.

## Step 4: Execution Sequence
Run the notebook cells in the following order:
1.  **Clone & Install**: Clones the repo and installs `ultralytics` and other dependencies.
2.  **Download Dataset**: Downloads and organizes the Moon crater data.
3.  **Train Stage 1**: Executes `train_yolov8.py`. (Expected duration: ~15-20 mins on T4).
4.  **Crop & Categorize**: Runs `crop_craters.py` to prepare Stage 2 data.
5.  **Train Stage 2**: Executes `train_classifier.py` (ResNet-50). (Expected duration: ~10 mins on T4).

## Step 5: Packaging Results
The final cell in the notebook zips the `models/` directory:
1.  Download `CraterNetX_Results.zip`.
2.  Unzip locally into your project's `models/` directory.

## Step 6: Local Validation
Run the Streamlit demo to verify the models:
```bash
streamlit run app/main.py
```
---
*Note: If training on a local GPU, ensure CUDA 11.x+ and cuDNN are correctly configured.*
