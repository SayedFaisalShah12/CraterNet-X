import streamlit as st
import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from inference.predict import CraterNetX

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CraterNet-X | Planetary Analysis",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    /* Styling for the main header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(#f8f9fa, #adb5bd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    /* Stats card styling */
    .stats-card {
        background-color: #1e2130;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #343a40;
        transition: transform 0.3s;
    }
    .stats-card:hover {
        transform: translateY(-5px);
        border-color: #4dabf7;
    }
    
    /* Metrics numbers */
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #4dabf7;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #ced4da;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.image("https://img.icons8.com/wired/128/ffffff/moon.png", width=100)
    st.title("Settings")
    st.markdown("---")
    
    st.subheader("Model Configuration")
    conf_threshold = st.slider("Detection Confidence", 0.05, 1.0, 0.15, help="Low confidence shows more candidates, high confidence shows only certain detections.")
    
    st.markdown("---")
    st.subheader("Project Info")
    st.info("CraterNet-X uses a two-stage Detection → Classification pipeline powered by YOLOv8 and ResNet-50.")
    st.write("v1.0.0 | Planetary AI Research")

# --- MODEL LOADING ---
@st.cache_resource
def load_pipeline():
    det_model = project_root / 'models' / 'stage1_detection' / 'moon_crater_yolov8' / 'weights' / 'best.pt'
    cls_model = project_root / 'models' / 'stage2_classification' / 'crater_classifier_resnet50.pt'
    
    if det_model.exists() and cls_model.exists():
        return CraterNetX(str(det_model), str(cls_model))
    return None

pipeline = load_pipeline()

# --- MAIN INTERFACE ---
st.markdown('<div class="main-header">🌑 CraterNet-X</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Two-Stage Planetary Surface Analysis & Crater Classification</div>', unsafe_allow_html=True)

if pipeline is None:
    st.error("🛑 Models not found! Please ensure training is complete or weights are in the `models/` directory.")
    st.stop()

# --- UPLOAD SECTION ---
uploaded_file = st.file_uploader("Unleash the pipeline - Upload a lunar surface image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save temp image
    temp_dir = project_root / 'app' / 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = temp_dir / uploaded_file.name
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create columns for layout
    col_input, col_output = st.columns(2)

    with col_input:
        st.subheader("Input Surface")
        st.image(uploaded_file, use_container_width=True)

    with col_output:
        st.subheader("CraterNet-X Vision")
        # Run Inference
        with st.spinner("Executing Deep Learning Pipeline..."):
            start_time = time.time()
            results, img0 = pipeline.predict(str(temp_path), conf=conf_threshold)
            inference_time = time.time() - start_time
            
            # Visualize
            res_img = pipeline.visualize(str(temp_path), results)
            res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            st.image(res_img_rgb, use_container_width=True)

    # --- RESULTS SUMMARY ---
    st.markdown("---")
    st.header("📊 Analysis Report")
    
    if results:
        counts = {'small': 0, 'medium': 0, 'large': 0}
        for r in results:
            counts[r['size_class']] += 1
        
        # Display cards
        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1:
            st.markdown(f'<div class="stats-card"><div class="metric-val">{len(results)}</div><div class="metric-label">Total Craters</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stats-card"><div class="metric-val">{counts["small"]}</div><div class="metric-label">Small</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stats-card"><div class="metric-val">{counts["medium"]}</div><div class="metric-label">Medium</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stats-card"><div class="metric-val">{counts["large"]}</div><div class="metric-label">Large</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="stats-card"><div class="metric-val">{inference_time:.2f}s</div><div class="metric-label">Speed</div></div>', unsafe_allow_html=True)

        # Tab view for data
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📋 Detection Log", "📈 Size Distribution"])
        
        with tab1:
            df = pd.DataFrame(results)
            if not df.empty:
                df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                st.dataframe(df, use_container_width=True)
        
        with tab2:
            chart_data = pd.DataFrame({
                'Size Class': ['Small', 'Medium', 'Large'],
                'Count': [counts['small'], counts['medium'], counts['large']]
            })
            st.bar_chart(chart_data.set_index('Size Class'))

    else:
        st.info("No craters detected in this region with current settings. Try lowering the confidence threshold.")

else:
    # Landing page info
    st.markdown("---")
    st.info("💡 **Getting Started:** Upload a lunar crop or high-resolution surface image to begin analysis.")
    
    st.markdown("""
    ### Pipeline Architecture
    1. **Stage 1: YOLOv8-Nano**
       Optimal detection of geological features using normalized bounding box coordinates.
    2. **Stage 2: ResNet-50 Classifier**
       High-precision classification focusing on crater morphology and relative area.
    """)
