#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:29:10 2025

@author: harikartha
"""

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ===== Custom CSS for Light Theme =====
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 6px;
        padding: 8px 16px;
        border: none;
        font-weight: 500;
    }
    .stFileUploader>div>div>div>button {
        background-color: #3498db;
        color: white;
    }
    .stAlert {
        background-color: #e3f2fd;
        border-left: 4px solid #3498db;
    }
    .footer {
        font-size: 12px;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        color: #7f8c8d;
        border-top: 1px solid #e0e0e0;
    }
    .result-container {
        border-radius: 8px;
        padding: 1.5rem;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== Page Setup =====
st.set_page_config(
    page_title="Pallet Detection Pro",
    page_icon="📦",
    layout="wide"
)

# ===== Header with Logo =====
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("logo final.png", width=600)  # Make sure logo.png is in your project folder
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0.5rem;'>Pallet Detection System</h1>",
        unsafe_allow_html=True
    )


# ===== Sidebar for Upload =====
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Detection Settings**")
    confidence = st.slider("Confidence threshold", 0.1, 1.0, 0.25)

# Load model (keep at bottom to avoid reloading)
@st.cache_resource
def load_model():
    return YOLO("Final_pallet_detection_model.pt")

model = load_model()


# ===== Main Content Area =====
if uploaded_file:
    # Original Image Display
    with st.container():
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    # Detection Results
    with st.spinner("🔍 Detecting pallets..."):
        results = model.predict(image, conf=confidence)
        result_image = results[0].plot()
        
        with st.container():
            st.subheader("Detection Results")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(result_image, use_container_width=True)
            
            with col2:
                # Count pallets
                names = model.names
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                pallet_count = sum(1 for cid in class_ids if names[cid].lower() == "pallet")
                
                st.metric(
                    label="Total Pallets Detected",
                    value=pallet_count,
                    help=f"Confidence threshold: {confidence*100}%"
                )
                
                if pallet_count > 0:
                    st.success("✅ Detection successful!")
                else:
                    st.warning("⚠️ No pallets detected")

# ===== Footer =====
st.markdown(
    """
    <div class="footer">
        Pallet Detection App – Powered by YOLOv8
    </div>
    """,
    unsafe_allow_html=True
)













