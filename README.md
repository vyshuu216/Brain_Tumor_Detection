
# 🧠 Brain Tumor Detection Using Deep Learning and Transfer Learning

## 📌 Project Overview

Brain tumors are among the most critical neurological diseases, where early detection plays a vital role in improving patient survival and treatment planning. Manual examination of MRI scans is time-consuming and requires expert radiologists.

This project presents an **End-to-End Brain Tumor Detection System** using **Deep Learning** and **Transfer Learning**. The system automatically classifies MRI brain scans into **Tumor** or **No Tumor** using the **MobileNetV2** architecture and provides prediction confidence through a user-friendly Flask web application.

---

# 🎯 Objectives

- Detect brain tumors from MRI images automatically.
- Improve diagnosis speed and accuracy.
- Reduce manual effort for radiologists.
- Develop a lightweight and deployable AI solution.
- Build an interactive web application for predictions.

---

# ✨ Features

- Brain MRI Classification
- MobileNetV2 Transfer Learning
- Image Preprocessing
- Data Augmentation
- Binary Classification (Tumor / No Tumor)
- Flask Web Application
- Upload MRI Images
- Real-time Prediction
- Confidence Score Display
- Model Saving (.keras)
- Confusion Matrix
- Precision
- Recall
- F1 Score
- Docker Support

---

# 🗂 Dataset

Dataset Name:

**Brain MRI Images for Brain Tumor Detection**

Source:

https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

Dataset contains:

- Tumor Images (Yes): **155**
- No Tumor Images (No): **98**

Total Images:

**253 MRI Images**

---

# 🛠 Technologies Used

## Programming Language

- Python

## Deep Learning

- TensorFlow
- Keras

## Model

- MobileNetV2

## Image Processing

- OpenCV
- NumPy
- Matplotlib

## Machine Learning

- Scikit-learn

## Deployment

- Flask
- Docker

---

# 🧠 Model Architecture

```
Input MRI Image
        │
        ▼
Image Preprocessing
        │
        ▼
Resize (224 × 224)
        │
        ▼
Normalization
        │
        ▼
Data Augmentation
        │
        ▼
MobileNetV2
        │
        ▼
Global Average Pooling
        │
        ▼
Dropout
        │
        ▼
Dense Layer (ReLU)
        │
        ▼
Dropout
        │
        ▼
Dense Layer (Sigmoid)
        │
        ▼
Prediction
```

---

# 📂 Project Structure

```
Brain_Tumor_Detection/

│
├── Brain_Tumor_Detection.ipynb
├── brain_tumor_model.keras
├── predict.py
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── templates/
│      └── index.html
│
├── static/
│      └── uploads/
│
└── Brain_Tumor_Dataset/
       ├── yes/
       └── no/
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/vyshuu216/Brain_Tumor_Detection.git
```

Go to project folder

```bash
cd Brain_Tumor_Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open browser

```
http://127.0.0.1:5000
```

---

# 🚀 Workflow

1. Upload MRI image.
2. Image preprocessing.
3. Resize to 224 × 224.
4. Normalize pixel values.
5. Pass image to MobileNetV2.
6. Extract deep features.
7. Perform binary classification.
8. Display prediction.
9. Display confidence score.

---

# 📊 Model Evaluation

Evaluation Metrics Used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# 💻 Web Application

The Flask application allows users to:

- Upload MRI images
- Predict Tumor/No Tumor
- Display confidence score
- Interactive user interface
- Easy deployment

---

# 📦 Docker Support

Build Docker Image

```bash
docker build -t brain-tumor-app .
```

Run Container

```bash
docker run -p 5000:5000 brain-tumor-app
```

---

# 📈 Future Enhancements

- Grad-CAM Visualization
- Multi-class Brain Tumor Classification
- MRI Report Generation
- Patient History Integration
- Cloud Deployment
- Mobile Application
- Explainable AI (XAI)

---

# 👨‍💻 Author

Brain Tumor Detection Using Deep Learning and Transfer Learning

Developed using:

- Python
- TensorFlow
- MobileNetV2
- Flask
- OpenCV
- Docker

---

# ⭐ Acknowledgement

Dataset:

Brain MRI Images for Brain Tumor Detection

https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

Frameworks:

- TensorFlow
- Keras
- Flask
- OpenCV
- Scikit-learn

# Brain_Tumor_Detection

