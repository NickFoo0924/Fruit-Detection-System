# 🍎 Fruit Detection System

A **Streamlit web application** for detecting and classifying fruits using **YOLOv8** and a **CNN model**. Users can upload images or use their camera to detect and classify fruits in real time.

---

## 1. About Dataset

- The system uses two types of models:
  1. **YOLOv8** – for detecting multiple fruits in an image.
  2. **CNN** – for classifying a single fruit into one of six categories:  
     - Apple  
     - Banana  
     - Grape  
     - Orange  
     - Pineapple  
     - Watermelon  

- Images should be clear and of reasonable size for better accuracy.

---

## 2. Repository Structure

```text
fruit-detection-system/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── weights/
|    └──  best.pt          # YOLOv8 model weights
└── cnn_fruit.h5           # CNN model 

---

## 3. Installation and Running

### Clone the repository:

```bash
git clone <your-repo-url>
cd fruit-detection-system

Install dependencies:
pip install -r requirements.txt

Add model:

You can either download the models manually or let the app download them automatically.

Manual download links:

YOLOv8 model (best.pt): Download here

CNN model (cnn_fruit.h5): Download here

Place these files in the weights/ folder. The app will create the folder if it doesn’t exist.

---

## 4. How Does the System Work

### YOLOv8 Detection Mode
1. The user uploads an image or takes a photo.  
2. YOLOv8 detects all fruits in the image.  
3. Detected fruits are highlighted with bounding boxes.  

### CNN Classification Mode
1. The user uploads an image or takes a photo.  
2. The image is resized to match the CNN input shape `(224x224x3)`.  
3. The CNN predicts which fruit is in the image.  
4. The result is displayed along with the prediction confidence.  

**Using the App:**
- The app will open in your browser.  
- Select a mode from the sidebar: **YOLOv8 Detection** or **CNN Classification**.  
- Upload an image or use the camera to see detection/classification results.  

---

## 5. Dependencies

- `streamlit`  
- `ultralytics`  
- `tensorflow` / `keras`  
- `opencv-python-headless`  
- `numpy`  
- `gdown` (optional, for downloading models programmatically)  


