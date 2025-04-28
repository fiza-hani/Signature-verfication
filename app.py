import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
from utils import preprocess_image, extract_features, compare_signatures, extract_advanced_features

# Sidebar Inputs
st.sidebar.title("📥 Upload Signatures")
genuine_file = st.sidebar.file_uploader("Upload Genuine Signature", type=["jpg", "png", "jpeg"])
test_file = st.sidebar.file_uploader("Upload Signature to Verify", type=["jpg", "png", "jpeg"])
threshold = st.sidebar.slider("Verification Threshold", 0, 5000, 1000)

st.markdown("""
<style>
.title-text {
    font-size:36px;
    font-weight:bold;
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-text'>Signature Verification System</div>", unsafe_allow_html=True)
st.markdown("""
This app verifies a signature against a genuine sample using advanced image processing techniques including contour analysis and shape descriptors.
""")

if genuine_file and test_file:
    # Load and preprocess
    genuine_img = np.array(Image.open(genuine_file).convert("RGB"))
    test_img = np.array(Image.open(test_file).convert("RGB"))
    genuine_pre = preprocess_image(genuine_img)
    test_pre = preprocess_image(test_img)

    # Extract advanced features
    feat1 = extract_advanced_features(genuine_pre)
    feat2 = extract_advanced_features(test_pre)

    # Split features
    area1, perimeter1, aspect_ratio1, hu_moments1 = feat1
    area2, perimeter2, aspect_ratio2, hu_moments2 = feat2

    # Combine features into vectors
    vec1 = np.concatenate([[area1, perimeter1, aspect_ratio1], hu_moments1])
    vec2 = np.concatenate([[area2, perimeter2, aspect_ratio2], hu_moments2])

    # Compare using Euclidean distance
    distance = compare_signatures(vec1, vec2)
    confidence = max(0, 100 - (distance / threshold * 100))

    # Show images
    st.subheader("🖼️ Original and Processed Images")
    col1, col2 = st.columns(2)
    col1.image(genuine_img, caption="Original Genuine", use_column_width=True)
    col2.image(test_img, caption="Original Test", use_column_width=True)

    col1, col2 = st.columns(2)
    col1.image(genuine_pre, caption="Processed Genuine", use_column_width=True)
    col2.image(test_pre, caption="Processed Test", use_column_width=True)

    # Result
    st.subheader("🔍 Results")
    st.write(f"Feature Distance: `{distance:.2f}`")
    if confidence > 80:
        st.success(f"High Confidence: {confidence:.2f}% ✅ Signature Verified")
    elif confidence > 50:
        st.warning(f"Medium Confidence: {confidence:.2f}% ⚠️ Verification Uncertain")
    else:
        st.error(f"Low Confidence: {confidence:.2f}% ❌ Signature Likely Forged")

    # Feature Difference Visualization with Plotly
    st.subheader("📊 Feature Comparison")
    features = ['Area', 'Perimeter', 'Aspect Ratio'] + [f'Hu{i+1}' for i in range(7)]
    differences = np.abs(vec1 - vec2)
    fig = px.bar(x=features, y=differences, color=features, title="Feature Differences Between Signatures")
    st.plotly_chart(fig)

    st.markdown("""
---
Made with ❤️ using Streamlit, OpenCV | [GitHub Repo](#)
""", unsafe_allow_html=True)
else:
    st.warning("Please upload both genuine and test signature images to begin.")
