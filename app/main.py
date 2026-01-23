import streamlit as st
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

# Add project root to path so we can import CraterNetX
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from inference.predict import CraterNetX

# Page config
st.set_page_config(page_title="CraterNet-X: Moon Crater Detection", layout="wide")

st.title("🌑 CraterNet-X: Automated Moon Crater Analysis")
st.markdown("""
This tool uses a two-stage deep learning pipeline:
1. **Detection (YOLOv8)**: Locates craters in lunar imagery.
2. **Classification (ResNet-50)**: Categorizes detected craters into Small, Medium, and Large.
""")

# Model Paths
DET_MODEL = project_root / 'models' / 'stage1_detection' / 'moon_crater_yolov8' / 'weights' / 'best.pt'
CLS_MODEL = project_root / 'models' / 'stage2_classification' / 'crater_classifier_resnet50.pt'

@st.cache_resource
def load_pipeline():
    if DET_MODEL.exists() and CLS_MODEL.exists():
        return CraterNetX(str(DET_MODEL), str(CLS_MODEL))
    return None

pipeline = load_pipeline()

if pipeline is None:
    st.warning("⚠️ Models not found. Please ensure both Stage-1 and Stage-2 training are complete.")
    st.info(f"Expected Detector: {DET_MODEL}")
    st.info(f"Expected Classifier: {CLS_MODEL}")
else:
    st.sidebar.header("Settings")
    conf_threshold = st.sidebar.slider("Detection Confidence", 0.05, 1.0, 0.15)
    
    uploaded_file = st.file_uploader("Upload a Moon Surface Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save temp image
        temp_dir = project_root / 'app' / 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run Inference
        with st.spinner("Analyzing Lunar Surface..."):
            results, img0 = pipeline.predict(str(temp_path), conf=conf_threshold)
            
        # Visualize
        res_img = pipeline.visualize(str(temp_path), results)
        res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, use_container_width=True)
            
        with col2:
            st.subheader("CraterNet-X Results")
            st.image(res_img_rgb, use_container_width=True)
            
        # Statistics
        st.subheader("Detection Summary")
        if results:
            counts = {'small': 0, 'medium': 0, 'large': 0}
            for r in results:
                counts[r['size_class']] += 1
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Craters", len(results))
            c2.metric("Small", counts['small'])
            c3.metric("Medium", counts['medium'])
            c4.metric("Large", counts['large'])
            
            # Details table
            st.dataframe(results)
        else:
            st.info("No craters detected with current confidence threshold.")

st.sidebar.markdown("---")
st.sidebar.write("Developed for Planetary Science Research")
