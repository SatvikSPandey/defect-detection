import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import io

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def run_inference(model, image_array, confidence):
    results = model.predict(image_array, conf=confidence, verbose=False)
    return results[0]

def draw_boxes(image_array, result):
    annotated = image_array.copy()
    class_names = result.names
    colors_map = {
        0: (255, 100, 100),
        1: (100, 255, 100),
        2: (100, 100, 255),
        3: (255, 255, 100),
        4: (255, 100, 255),
        5: (100, 255, 255),
    }
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls_id]} {conf:.2f}"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = colors_map.get(cls_id, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return annotated

def get_defect_summary(result):
    summary = {}
    class_names = result.names
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        conf = float(box.conf[0])
        if cls_name not in summary:
            summary[cls_name] = {"count": 0, "confidences": []}
        summary[cls_name]["count"] += 1
        summary[cls_name]["confidences"].append(round(conf, 3))
    return summary

def generate_pdf_report(summary, annotated_image_array, filename):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Industrial Defect Detection Report", styles['Title']))
    elements.append(Spacer(1, 0.4*cm))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Paragraph(f"Source file: {filename}", styles['Normal']))
    elements.append(Spacer(1, 0.6*cm))

    if not summary:
        elements.append(Paragraph("No defects detected.", styles['Normal']))
    else:
        elements.append(Paragraph("Defect Summary", styles['Heading2']))
        elements.append(Spacer(1, 0.3*cm))

        table_data = [["Defect Class", "Count", "Avg Confidence", "Max Confidence"]]
        for cls_name, info in summary.items():
            avg_conf = round(sum(info["confidences"]) / len(info["confidences"]), 3)
            max_conf = round(max(info["confidences"]), 3)
            table_data.append([cls_name, str(info["count"]), str(avg_conf), str(max_conf)])

        table = Table(table_data, colWidths=[5*cm, 3*cm, 4*cm, 4*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f2f2f2')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.6*cm))

    elements.append(Paragraph("Annotated Image", styles['Heading2']))
    elements.append(Spacer(1, 0.3*cm))

    img_pil = Image.fromarray(cv2.cvtColor(annotated_image_array, cv2.COLOR_BGR2RGB))
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    rl_image = RLImage(img_buffer, width=14*cm, height=10*cm)
    elements.append(rl_image)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- UI ---
st.title("🔍 Industrial Defect Detection")
st.markdown("Upload a steel surface image to detect and classify manufacturing defects using YOLOv8.")

with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 0.95, 0.25, 0.05)
    st.markdown("---")
    st.markdown("**Model:** YOLOv8s fine-tuned on NEU Steel Defect Dataset")
    st.markdown("**Classes:** crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches")
    st.markdown("**mAP50:** 77.6%")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    model = load_model()

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = run_inference(model, image_bgr, confidence)
    annotated_bgr = draw_boxes(image_bgr, result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("Detected Defects")
        st.image(annotated_rgb, use_container_width=True)

    summary = get_defect_summary(result)

    st.subheader("📊 Detection Summary")
    if not summary:
        st.success("✅ No defects detected above the confidence threshold.")
    else:
        total = sum(v["count"] for v in summary.values())
        st.error(f"⚠️ {total} defect(s) detected across {len(summary)} class(es)")
        cols = st.columns(len(summary))
        for i, (cls_name, info) in enumerate(summary.items()):
            with cols[i]:
                avg_conf = sum(info["confidences"]) / len(info["confidences"])
                st.metric(label=cls_name, value=f'{info["count"]} found',
                          delta=f'avg conf: {avg_conf:.2f}')

    st.subheader("📄 Download Report")
    col_img, col_pdf = st.columns(2)

    with col_img:
        img_pil = Image.fromarray(annotated_rgb)
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='PNG')
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=img_buffer.getvalue(),
            file_name=f"defect_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )

    with col_pdf:
        pdf_buffer = generate_pdf_report(summary, annotated_bgr, uploaded_file.name)
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_buffer,
            file_name=f"defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
else:
    st.info("👆 Upload an image to begin defect detection.")