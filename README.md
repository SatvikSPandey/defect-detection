# Industrial Defect Detection — YOLOv8

A production-grade industrial defect detection system that identifies and localises surface defects in steel manufacturing parts using YOLOv8 object detection. Detects 6 defect classes with bounding boxes, confidence scores, and generates a downloadable PDF quality control report.

**Live Demo:** [defect-detection-satvik.streamlit.app](https://defect-detection-satvik.streamlit.app)

---

## What It Does

- Accepts steel surface images via drag-and-drop upload
- Runs YOLOv8s inference and draws bounding boxes around detected defects
- Displays a detection summary with defect counts and confidence scores per class
- Generates a downloadable PDF report — defect class, count, avg/max confidence, and annotated image embedded
- Exports the annotated image as a PNG download
- Adjustable confidence threshold slider for sensitivity tuning

---

## Defect Classes

| Class | Description |
|---|---|
| crazing | Network of fine surface cracks |
| inclusion | Foreign material embedded in surface |
| patches | Discoloured surface regions |
| pitted_surface | Small pits or holes |
| rolled-in_scale | Scale pressed into the surface |
| scratches | Linear surface damage |

---

## Model Performance

Trained for 50 epochs on the NEU Steel Surface Defect Dataset (1,799 images) using transfer learning from pretrained YOLOv8s weights.

| Class | mAP50 |
|---|---|
| patches | 95.3% |
| scratches | 90.8% |
| inclusion | 87.2% |
| pitted_surface | 81.9% |
| rolled-in_scale | 66.2% |
| crazing | 44.3% |
| **Overall** | **77.6%** |

Crazing is the hardest class — fine surface cracks with subtle visual texture differences, consistent with published benchmarks on this dataset.

---

## Architecture

Input Image
│
▼
YOLOv8s (fine-tuned)
│
▼
Bounding Boxes + Class Labels + Confidence Scores
│
├──▶ Annotated Image Display
├──▶ Detection Summary (per-class count + confidence)
├──▶ PNG Download
└──▶ PDF Quality Control Report

**Why YOLOv8:** Single-stage detector — one forward pass produces all bounding boxes simultaneously. Faster than two-stage detectors (e.g. Faster R-CNN), suitable for real-time manufacturing inspection pipelines.

**Why transfer learning:** YOLOv8s pretrained on COCO already understands edges, textures, and shapes. Fine-tuning on domain-specific defect images converges faster and achieves higher accuracy than training from scratch.

---

## Tech Stack

- **Model:** YOLOv8s (Ultralytics)
- **Training:** Python, CUDA (NVIDIA GPU)
- **Dataset:** NEU Steel Surface Defect Dataset via Roboflow (1,799 images, 6 classes)
- **Frontend:** Streamlit
- **Image processing:** OpenCV, Pillow
- **Report generation:** ReportLab (PDF)
- **Deployment:** Streamlit Cloud

---

## Run Locally

```bash
git clone https://github.com/SatvikSPandey/defect-detection.git
cd defect-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

defect-detection/
├── app.py                  # Streamlit app (inference + PDF report)
├── train.py                # Training script (local GPU)
├── models/
│   └── best.pt             # Fine-tuned YOLOv8s weights
├── requirements.txt
├── packages.txt            # Streamlit Cloud system dependencies
└── README.md

---

## Sample Output

Upload any steel surface image and the model will detect defects, draw labelled bounding boxes, and generate a PDF quality control report showing defect class, count, and confidence scores — mirroring what a real manufacturing inspection system would produce.

---

## Acknowledgements

Dataset: [NEU Steel Surface Defect Dataset](https://universe.roboflow.com/neudatasetoriginal/neu-steel-defect-dataset) — CC BY 4.0

---

*Built by [Satvik Pandey](https://github.com/SatvikSPandey)*