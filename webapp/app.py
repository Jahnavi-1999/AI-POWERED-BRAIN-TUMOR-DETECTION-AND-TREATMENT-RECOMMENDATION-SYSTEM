import sys
import os
import tempfile
import base64
import re
from datetime import datetime

# ✅ Add root directory to sys.path to allow sibling imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from fpdf import FPDF

# ✅ Import from sibling folders
from models.hybrid_vgg16_resnet50 import build_hybrid_model
from utils.gradcam import generate_gradcam

# ------------------- Streamlit UI Configuration -------------------
st.set_page_config(page_title="Brain Tumor Detector", layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f5f6fa;
            color: #1c1c1c;
        }
        h1 {
            font-size: 2.8rem;
            font-weight: 800;
            color: #2f3640;
            text-align: center;
            margin-bottom: 10px;
        }
        h2 {
            font-size: 1.8rem;
            color: #0097e6;
            margin-top: 40px;
            border-bottom: 1px solid #ccc;
            padding-bottom: 6px;
        }
        .stTabs [role="tab"] {
            background: #dcdde1;
            color: #2f3640;
            font-weight: bold;
            border-radius: 8px 8px 0 0;
            padding: 10px;
            margin-right: 5px;
        }
        .stTabs [aria-selected="true"] {
            background: #00a8ff;
            color: white;
        }
        .stButton > button {
            background-color: #273c75;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 16px;
        }
        .stButton > button:hover {
            background-color: #192a56;
        }
        .image-row {
            display: flex;
            justify-content: space-around;
            align-items: center;
            gap: 96px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Brain Tumor Detection & Symptom-Based Recommendation System")

# ------------------- Load Model -------------------
model = build_hybrid_model()
model.load_weights(os.path.join(root_dir, "hybrid_model_weights.h5"))
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ------------------- PDF Generator -------------------
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def generate_pdf(tumor, symptom_tumor, treatment_text, patient_name, patient_id):
    logo_path = os.path.join(current_dir, "logo.png")
    class PDF(FPDF):
        def header(self):
            try:
                self.image(logo_path, x=10, y=8, w=18)
            except:
                pass
            self.set_font("Arial", 'B', 16)
            self.set_text_color(0, 70, 140)
            self.cell(0, 10, "MEDICAL REPORT", border=False, ln=True, align='C')
            self.ln(10)

    pdf = PDF()
    pdf.add_page()
    pdf.set_fill_color(0, 0, 0)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "SECTION 1: PATIENT'S PARTICULARS", ln=True, fill=True)

    pdf.set_text_color(0)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(60, 10, "Full name of patient:", border=1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, remove_emojis(patient_name), border=1, ln=True)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(60, 10, "Patient ID:", border=1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, remove_emojis(patient_id), border=1, ln=True)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(60, 10, "Tumor Prediction:", border=1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 10, remove_emojis(tumor.capitalize()), border=1, ln=True)

    if symptom_tumor:
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(60, 10, "Symptom Diagnosis:", border=1)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 10, remove_emojis(symptom_tumor.capitalize()), border=1, ln=True)

    pdf.ln(4)
    pdf.set_fill_color(0, 0, 0)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "SECTION 2: TREATMENT RECOMMENDATION", ln=True, fill=True)

    pdf.set_text_color(0)
    pdf.set_font("Arial", '', 11)
    clean_text = remove_emojis(treatment_text)
    pdf.multi_cell(0, 8, clean_text, border=1)

    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'C')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        pdf.output(temp_file.name)
        with open(temp_file.name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    return base64_pdf

# ------------------- Static Data -------------------
treatment_map = {
    "glioma": "Immediate neuro-oncological evaluation is recommended.\n- MRI with contrast\n- Biopsy\n- Surgery, radiation, chemotherapy",
    "meningioma": "Often slow-growing, may require observation.\n- Regular MRI monitoring\n- Surgery if symptomatic",
    "pituitary": "Can affect hormones.\n- Blood tests and MRI\n- Hormone therapy or surgery",
    "notumor": "No tumor detected.\n- Regular checkups and monitoring"
}

symptom_db = {
    "Persistent headache": ("glioma", 2),
    "Seizures": ("glioma", 3),
    "Blurred vision": ("pituitary", 2),
    "Memory issues": ("meningioma", 2),
    "Difficulty speaking": ("glioma", 2),
    "Loss of balance": ("meningioma", 3),
    "Nausea": ("pituitary", 1),
    "Weakness in limbs": ("glioma", 3),
    "Menstrual irregularities": ("pituitary", 2),
    "Hearing problems": ("meningioma", 1)
}

def get_timeline(tumor_type):
    return {
        "glioma": ["MRI with contrast", "Biopsy", "Surgery", "Radiation", "Follow-up imaging"],
        "meningioma": ["MRI detection", "Monitor or Surgery", "Yearly MRI check"],
        "pituitary": ["Hormone testing", "MRI", "Hormone meds", "Surgery if needed"],
        "notumor": ["Routine checkups", "Monitor symptoms"]
    }.get(tumor_type, [])

# ------------------- Tabs -------------------
tab1, tab2 = st.tabs(["🧠 MRI Detection", "🧰 Symptom-Based Analysis"])

# -------- MRI Detection --------
with tab1:
    st.header("📸 MRI Image Upload & Detection")
    with st.form("patient_info_form1"):
        name = st.text_input("👤 Patient Name")
        pid = st.text_input("🆔 Patient ID")
        submitted = st.form_submit_button("Submit Info")
    if not name or not pid:
        st.warning("⚠️ Please enter both patient name and ID.")
        st.stop()

    file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])
    if file:
        try:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_norm = img_resized.astype(np.float32) / 255.0
            img_input = np.expand_dims(img_norm, axis=0)

            st.markdown('<div class="image-row">', unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.image(img_resized, caption="Uploaded MRI", width=300)

            pred = model.predict([img_input, img_input])
            raw_pred = class_names[np.argmax(pred)]
            tumor_pred = raw_pred.replace('_tumor', '')
            st.success(f"🧠 Predicted Tumor Type: **{tumor_pred.capitalize()}**")

            if st.checkbox("🔍 Show Grad-CAM"):
                heatmap = generate_gradcam(model, img_norm, layer_name="block5_conv3")

                with col2:
                    st.image(heatmap, caption="Grad-CAM Heatmap", width=300)

            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("🎯 MRI-Based Treatment Recommendation")
            treatment = treatment_map.get(tumor_pred, "⚠️ No treatment information available.")
            st.markdown(treatment, unsafe_allow_html=True)

            st.subheader("🗓 Suggested Treatment Timeline")
            for step in get_timeline(tumor_pred):
                st.markdown(f"- {step}")

            base64_pdf = generate_pdf(tumor_pred, None, treatment, name, pid)
            st.download_button("📄 Download PDF Report", base64.b64decode(base64_pdf), "tumor_report.pdf")

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

# -------- Symptom-Based Analysis --------
with tab2:
    st.header("🧰 Symptom-Based Assessment & Recommendation")
    with st.form("patient_info_form2"):
        name2 = st.text_input("👤 Patient Name", key="symptom_name")
        pid2 = st.text_input("🆔 Patient ID", key="symptom_pid")
        submitted2 = st.form_submit_button("Submit Info")
    if not name2 or not pid2:
        st.warning("⚠️ Please enter both patient name and ID.")
        st.stop()

    selected_symptoms = st.multiselect("Select your symptoms:", list(symptom_db.keys()))
    if st.button("🧪 Analyze Symptoms"):
        if not selected_symptoms:
            st.info("✅ No symptoms selected.")
        else:
            scores = {"glioma": 0, "meningioma": 0, "pituitary": 0}
            for symptom in selected_symptoms:
                tumor, intensity = symptom_db[symptom]
                scores[tumor] += intensity

            likely = max(scores, key=scores.get)
            total_score = scores[likely]

            st.subheader(f"🧠 Likely Tumor Type from Symptoms: **{likely.capitalize()}**")
            st.write(f"🧪 Symptom Intensity Score: **{total_score}**")

            if total_score >= 6:
                st.warning("⚠️ High symptom intensity. Urgent screening recommended.")
            elif total_score >= 3:
                st.info("🔍 Moderate symptoms. Consider seeing a specialist.")
            else:
                st.success("✅ Mild symptoms. Monitor for changes.")

            st.subheader("🌟 Symptom-Based Treatment Recommendation")
            st.markdown(treatment_map[likely], unsafe_allow_html=True)

            st.subheader("🗓 Suggested Treatment Timeline")
            for step in get_timeline(likely):
                st.markdown(f"- {step}")

            base64_pdf = generate_pdf("N/A", likely, treatment_map[likely], name2, pid2)
            st.download_button("📄 Download PDF Report (Symptoms)", base64.b64decode(base64_pdf), "symptom_report.pdf")

st.markdown("---")
st.write("👋 Thanks for using the Brain Tumor Symptom Analyzer!")
