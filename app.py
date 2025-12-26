import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Model Load
model = YOLO('best.pt')

st.set_page_config(page_title="Fish Disease Detector", layout="centered")
st.title("🐟 Fish Disease Detection")
st.write("Take a photo or upload an image to detect diseases.")

# Mobile-er camera ba gallery use korar jonno uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner('Detecting...'):
        results = model(image)
        res_plotted = results[0].plot()
        res_image = Image.fromarray(res_plotted[:, :, ::-1])

    st.success("Analysis Done!")
    st.image(res_image, caption='Detected Results', use_container_width=True)

    # Class names summary
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        st.warning(f"Detected: {label} ({conf:.2f})")
