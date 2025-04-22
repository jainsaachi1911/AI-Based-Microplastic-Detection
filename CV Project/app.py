import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import os
import joblib
import json
from ultralytics import YOLO
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# --- Load YOLO model and classifier ---
yolo_model = YOLO('best.pt')
clf = joblib.load('pollution_classifier_model.joblib')
le = joblib.load('label_encoder.joblib')

# --- Feature extraction function from YOLO labels ---
def extract_features_from_results(results):
    boxes = results[0].boxes
    count = len(boxes)
    sizes = []
    aspect_ratios = []

    for box in boxes:
        xywh = box.xywh[0].cpu().numpy()
        w, h = xywh[2], xywh[3]
        sizes.append(w * h)
        if h > 0:
            aspect_ratios.append(w / h)

    avg_size = np.mean(sizes) if sizes else 0
    std_size = np.std(sizes) if sizes else 0
    max_size = np.max(sizes) if sizes else 0
    avg_aspect = np.mean(aspect_ratios) if aspect_ratios else 0
    std_aspect = np.std(aspect_ratios) if aspect_ratios else 0

    return pd.DataFrame([{
        'count': count,
        'avg_size': avg_size,
        'std_size': std_size,
        'max_size': max_size,
        'avg_aspect': avg_aspect,
        'std_aspect': std_aspect
    }])

st.markdown(
    """
    <style>
    .main .block-container {
        padding-left: 10%;
        padding-right: 10%;
        max-width: 1100px;
        margin-left: auto;
        margin-right: auto;
    }

    html, body, div, p, label, span {
        font-size: 13px !important;
        line-height: 1.4;
    }

    h1 {
        font-size: 28px !important;
    }
    h2, h3, h4 {
        font-size: 20px !important;
    }

    .stFileUploader label, .stFileUploader span,
    .stMetric label, .stMetric div {
        font-size: 13px !important;
    }

    div[data-testid="stDataFrameContainer"] {
        font-size: 12px !important;
    }

    .block-container > div {
        margin-bottom: 1rem;
    }

    .element-container {
        padding: 0.3rem !important;
    }

    img {
        max-width: 100%;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Streamlit UI ---
st.title("Microplastic Detection and Water Quality Analysis")
st.write("Upload a microscopic image of water and receive pollution classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    # --- Run YOLO detection ---
    results = yolo_model.predict(image_path, save=False, conf=0.25)
    image = results[0].plot()

    # --- Display side-by-side images ---
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.image(image_path, caption="Uploaded Image", width=200)
    with col2:
        st.image(image, caption="Detected Microplastics", width=200)

    # --- Extract features and predict ---
    features_df = extract_features_from_results(results)
    prediction = clf.predict(features_df)[0]
    confidence = clf.predict_proba(features_df).max()
    label = le.inverse_transform([prediction])[0]

    # --- Display results ---
    col3, col4 = st.columns([1, 1])
    with col3:
        st.metric("Pollution Level", label)
    with col4:
        st.write(f"**Confidence:** {confidence:.2f}")

    st.subheader("Extracted Features")
    st.dataframe(features_df, use_container_width=True)

    st.info(f"Explanation: Detected {features_df['count'].values[0]} particles with an average size of {features_df['avg_size'].values[0]:.5f}. Classified as '{label}'.")