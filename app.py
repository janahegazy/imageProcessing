import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="Image Enhancer", layout="wide", initial_sidebar_state="expanded")

# CSS styling
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
        }
        .description {
            font-size: 20px;
            color: #34495E;
            text-align: center;
        }
        .filter-title {
            color: #FF69B4;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .apply-button > div > button {
            background-color: #FF69B4;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🎨 Image Enhancement Website</p>', unsafe_allow_html=True)

# Description
st.markdown(
    '<p class="description">Upload an image and enhance it using various filters. Adjust their intensity using sliders for a more customized result!</p>',
    unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📥 Upload an image", type=["jpg", "jpeg", "png"])
selected_filters = []

# Filter functions
def gaussian_smoothing(img, intensity):
    return cv2.GaussianBlur(img, (intensity, intensity), 0)

def unsharp_masking(img, intensity):
    blurred = cv2.GaussianBlur(img, (intensity, intensity), 0)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

def power_law_transform(img, gamma):
    return np.array(255 * (img / 255) ** gamma, dtype='uint8')

def histogram_processing(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    elif len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def threshold_segmentation(img, threshold_value):
    _, thresholded = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

def bilinear_interpolation(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

# Display logic
if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Resize image for display if too large
    max_display_size = 700
    h, w = image_np.shape[:2]
    if max(h, w) > max_display_size:
        scale = max_display_size / max(h, w)
        image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)))

    st.image(image_np, caption="Original Image", use_container_width=True)

    st.markdown("### 🔍 <span class='filter-title'>Choose filters to apply:</span>", unsafe_allow_html=True)

    # 2 Columns layout
    col1, col2 = st.columns(2)

    # Filters UI
    with col1:
        st.markdown("<div class='filter-title'>Unsharp Masking</div>", unsafe_allow_html=True)
        intensity1 = st.slider("Unsharp Intensity", 1, 15, 5, step=2)
        preview1 = unsharp_masking(image_np, intensity1)
        st.image(preview1, caption="Unsharp Masking Preview", use_container_width=True)
        if st.checkbox("Unsharp Masking"):
            selected_filters.append(("unsharp_masking", intensity1))

        st.markdown("<div class='filter-title'>Histogram Equalization</div>", unsafe_allow_html=True)
        preview4 = histogram_processing(image_np)
        st.image(preview4, caption="Histogram Equalization Preview", use_container_width=True)
        if st.checkbox("Histogram Equalization"):
            selected_filters.append(("histogram_equalization", None))

        st.markdown("<div class='filter-title'>Threshold Segmentation</div>", unsafe_allow_html=True)
        thresh_val = st.slider("Threshold", 0, 255, 127)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        preview5 = threshold_segmentation(gray, thresh_val)
        st.image(preview5, caption="Threshold Preview", use_container_width=True)
        if st.checkbox("Threshold Segmentation"):
            selected_filters.append(("threshold_segmentation", thresh_val))

    with col2:
        st.markdown("<div class='filter-title'>Power Law Transform</div>", unsafe_allow_html=True)
        gamma = st.slider("Gamma", 0.1, 3.0, 1.0, step=0.1)
        preview2 = power_law_transform(image_np, gamma)
        st.image(preview2, caption="Power Law Preview", use_container_width=True)
        if st.checkbox("Power Law Transform"):
            selected_filters.append(("power_law_transform", gamma))

        st.markdown("<div class='filter-title'>Bilinear Interpolation</div>", unsafe_allow_html=True)
        scale = st.slider("Scale %", 10, 200, 100, step=10)
        preview6 = bilinear_interpolation(image_np, scale)
        st.image(preview6, caption="Bilinear Interpolation Preview", use_container_width=True)
        if st.checkbox("Bilinear Interpolation"):
            selected_filters.append(("bilinear_interpolation", scale))

    # Apply button
    if st.button("✅ Apply Filters", key="apply", use_container_width=True):
        result = image_np.copy()
        for filt, val in selected_filters:
            if filt == "unsharp_masking":
                result = unsharp_masking(result, val)
            elif filt == "power_law_transform":
                result = power_law_transform(result, val)
            elif filt == "histogram_equalization":
                result = histogram_processing(result)
            elif filt == "threshold_segmentation":
                result = threshold_segmentation(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), val)
            elif filt == "bilinear_interpolation":
                result = bilinear_interpolation(result, val)

        st.markdown("### 🖼️ <span class='filter-title'>Result Image:</span>", unsafe_allow_html=True)
        st.image(result, width=400)  # Reduced size
