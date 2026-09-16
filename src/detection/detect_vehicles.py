"""
Step 1: Vehicle detection with YOLOv8.
Loads a pretrained YOLOv8 model (COCO classes already include
car/bus/truck/motorbike) and runs detection on an image or video.
"""
from ultralytics import YOLO

VEHICLE_CLASSES = {
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    return YOLO(weights)


def detect(model: YOLO, source: str, conf: float = 0.25, save: bool = True):
    results = model.predict(source=source, conf=conf, save=save, project="outputs", name="detect")
    return results


def summarize(results) -> dict:
    counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                counts[VEHICLE_CLASSES[cls_id]] += 1
    return counts


if __name__ == "__main__":
    model = load_model()
    results = detect(model, source="data/raw/sample_traffic.jpg")
    counts = summarize(results)
    print("Vehicle counts:", counts)
