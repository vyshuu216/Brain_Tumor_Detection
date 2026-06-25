# 🧠 Brain Tumor Detection Using Deep Learning and Transfer Learning

## 📌 Project Overview

Brain tumors are among the most critical neurological diseases, where early detection plays a vital role in improving patient survival and treatment planning. Manual examination of MRI scans is time-consuming and requires expert radiologists.

This project presents an **End-to-End Brain Tumor Detection System** using **Deep Learning**, **Transfer Learning**, and **Explainable AI (XAI)**. The system automatically classifies MRI brain scans into **Tumor** or **No Tumor** using the **MobileNetV2** architecture, provides confidence scores, and generates **Grad-CAM heatmaps** to visually explain which regions of the MRI influenced the prediction. The complete solution is deployed through a Flask web application and is Docker-ready.

---

# ✨ Features

* Brain MRI Classification
* MobileNetV2 Transfer Learning
* Image Preprocessing
* Data Augmentation
* Binary Classification (Tumor / No Tumor)
* Flask Web Application
* Upload MRI Images
* Real-time Prediction
* Confidence Score Display
* **Grad-CAM Visualization**
* Explainable AI (XAI)
* Confusion Matrix
* Precision
* Recall
* F1 Score
* Classification Report
* Model Saving (.keras)
* Docker Support

---

# 🧠 Explainable AI using Grad-CAM

One of the major features of this project is the implementation of **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

Grad-CAM provides visual explanations for the model's predictions by highlighting the regions of the MRI scan that contributed most to the classification.

### Workflow

```
MRI Image
      │
      ▼
MobileNetV2
      │
      ▼
Prediction
      │
      ▼
Gradient Computation
      │
      ▼
Activation Heatmap
      │
      ▼
Overlay on Original MRI
```

### Advantages

* Improves model interpretability.
* Helps doctors understand AI predictions.
* Increases confidence in model decisions.
* Supports Explainable Artificial Intelligence (XAI).

---

# 🚀 Complete Workflow

```
MRI Image Upload
        │
        ▼
Image Preprocessing
        │
        ▼
Resize (224×224)
        │
        ▼
Normalization
        │
        ▼
MobileNetV2 Transfer Learning
        │
        ▼
Prediction
        │
        ├──────────────► Confidence Score
        │
        ▼
Grad-CAM Generation
        │
        ▼
Overlay Heatmap
        │
        ▼
Display Results in Flask Web Application
```

---

# 📊 Model Evaluation

The trained model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report
* Prediction Confidence
* Grad-CAM Visual Validation

---

# 💻 Web Application

The Flask application provides:

* Upload MRI image
* Automatic preprocessing
* Tumor / No Tumor prediction
* Confidence score
* Original MRI display
* Grad-CAM heatmap visualization
* User-friendly interface

---

# 📸 Results

The project successfully detects brain tumors from MRI images and provides explainable predictions.

The output includes:

* Original MRI image
* Predicted class
* Confidence score
* Grad-CAM heatmap highlighting the suspected tumor region

Example:

```
Prediction : Tumor

Confidence : 98.74%

Original MRI      Grad-CAM
```

(Add screenshots inside the `screenshots/` folder and reference them here.)

---

# 📈 Future Enhancements

* Multi-class Brain Tumor Classification
* 3D MRI Volume Analysis
* MRI Report Generation
* Patient History Integration
* Cloud Deployment
* Mobile Application
* Medical Database Integration
* Improved Explainable AI Techniques
