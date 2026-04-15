from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="C:/Users/babaw/OneDrive/Desktop/work/Project12-Defect-Detection/data/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project="C:/Users/babaw/OneDrive/Desktop/work/Project12-Defect-Detection/runs",
        name="defect-detection",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True
    )

    print("Training complete!")
    print(f"Best model saved at: {results.save_dir}")