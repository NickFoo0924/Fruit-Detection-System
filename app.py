import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import os

# ======================
# Load Models
# ======================
yolo_model_path = "best.pt"
cnn_model_path = "weights/cnn_model.h5"
if not os.path.exists(cnn_model_path):
    os.makedirs("weights", exist_ok=True)
    url = "https://drive.google.com/uc?export=download&id=1rpszJBxSTXYG_Z8xe_I_eXQD32RM4IIz"
    gdown.download(url, cnn_model_path, quiet=False)

# Load YOLOv8 model
yolo_model = YOLO(yolo_model_path)

# Load CNN model
cnn_model = load_model(cnn_model_path)

# CNN class labels
cnn_labels = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"]

# ======================
# Sidebar Navigation
# ======================
st.sidebar.title("🍎 Fruit Detection System")
mode = st.sidebar.radio("🔎 Choose a Mode:", ["YOLOv8 Detection", "CNN Classification"])

# ======================
# YOLO Mode
# ======================
if mode == "YOLOv8 Detection":
    st.title("🍌 YOLOv8 Fruit Detection")
    st.write("Upload an image or take a snapshot with your camera to detect fruits.")

    # File uploader or Camera
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("Or take a picture 📸")

    frame = None
    if uploaded_file is not None:
        img_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    elif camera_file is not None:
        bytes_data = camera_file.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run YOLO detection
    if frame is not None:
        results = yolo_model(frame, conf=0.5)
        annotated_frame = results[0].plot()

        st.image(
            annotated_frame,
            channels="BGR",
            caption="✅ YOLOv8 Detected Fruits",
            use_column_width=True
        )

# ======================
# CNN Mode
# ======================
elif mode == "CNN Classification":
    st.title("🍊 CNN Fruit Classification")
    st.write("Upload an image or take a snapshot with your camera for classification.")

    # File uploader or Camera
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("Or take a picture 📸")

    frame = None
    if uploaded_file is not None:
        img_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    elif camera_file is not None:
        bytes_data = camera_file.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run CNN classification
    if frame is not None:
        resized = cv2.resize(frame, (128, 128))  # match CNN input shape
        arr = img_to_array(resized) / 255.0
        arr = np.expand_dims(arr, axis=0)

        pred = cnn_model.predict(arr)
        label = cnn_labels[np.argmax(pred)]
        confidence = np.max(pred)

        st.image(
            frame,
            channels="BGR",
            caption=f"✅ CNN Prediction: {label} ({confidence:.2f})",
            use_column_width=True
        )
