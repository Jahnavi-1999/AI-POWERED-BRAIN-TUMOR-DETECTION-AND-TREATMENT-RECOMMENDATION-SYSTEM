

# Brain Tumor Detection & Classification System

## Overview

This is a deep learning-based platform for accurate brain tumor detection and classification from MRI scans. The system utilizes a hybrid CNN model combining VGG16 and ResNet-50, along with EfficientNetB2 and Grad-CAM for explainability. A Streamlit-based web application enables users to upload scans, view classification results, and download personalized PDF reports.

---

## Key Features

* **Hybrid Deep Learning Model**: Combines VGG16 and ResNet-50 for improved accuracy and robustness.
* **Explainable AI**: Integrates Grad-CAM to highlight tumor regions influencing the model's decision.
* **Streamlit Web Interface**: Simple and intuitive interface for uploading images and viewing results.
* **Symptom-Based Screening**: Optional user input allows preliminary assessment based on symptoms.
* **Automated PDF Reports**: Generates downloadable reports with predictions, Grad-CAM visualizations, and patient details.

---

## Technology Stack

* **Deep Learning Models**: VGG16, ResNet-50, EfficientNetB2
* **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping)
* **Frontend**: Streamlit
* **Backend**: Python, TensorFlow/Keras, OpenCV
* **PDF Generation**: ReportLab, FPDF, or PyMuPDF

---

## How It Works

1. The user uploads a contrast-enhanced MRI scan via the web application.
2. The image is preprocessed and passed through the hybrid classification model.
3. The system predicts the tumor type: Glioma, Meningioma, Pituitary, or No Tumor.
4. Grad-CAM highlights the critical regions used by the model for decision-making.
5. A PDF report is generated with results, visual explanation, and patient information.

---

**For inquiries or collaborations, contact:**
**[ashrithsambaraju@gmail.com](mailto:ashrithsambaraju@gmail.com)**

