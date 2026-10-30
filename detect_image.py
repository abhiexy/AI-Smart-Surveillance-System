
import cv2
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================

MODEL_NAME = "yolo26n.pt"
IMAGE_PATH = "bus.jpg"
CONFIDENCE = 0.35

# COCO class IDs
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# LOAD MODEL
# =========================

print("Loading YOLO model...")
model = YOLO(MODEL_NAME)

# =========================
# DETECTION
# =========================

results = model.predict(
    source=IMAGE_PATH,
    classes=list(TARGET_CLASSES.keys()),
    conf=CONFIDENCE,
    imgsz=640,
    device="cpu",
    verbose=False
)

detections = []

# =========================
# PROCESS RESULTS
# =========================

for result in results:

    annotated_image = result.plot()

    for box in result.boxes:

        class_id = int(box.cls[0])
        object_name = TARGET_CLASSES[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        record = {
            "object": object_name,
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "bounding_box": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            },
            "timestamp": datetime.now().isoformat()
        }

        detections.append(record)

        print(
            f"{object_name} | "
            f"Confidence: {confidence:.2%} | "
            f"Coordinates: {x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}"
        )

    # Save annotated image
    cv2.imwrite(
        str(OUTPUT_DIR / "person_vehicle_detection.jpg"),
        annotated_image
    )

# =========================
# OBJECT COUNT
# =========================

counts = Counter(item["object"] for item in detections)

print("\n========== OBJECT SUMMARY ==========")

for name in TARGET_CLASSES.values():
    print(f"{name}: {counts.get(name, 0)}")

print("Total objects:", len(detections))

# =========================
# SAVE JSON
# =========================

output_data = {
    "camera_id": "IMAGE_TEST",
    "timestamp": datetime.now().isoformat(),
    "total_objects": len(detections),
    "counts": dict(counts),
    "detections": detections
}

with open(OUTPUT_DIR / "person_vehicle_results.json", "w") as file:
    json.dump(output_data, file, indent=4)

print("\nDetection completed successfully!")
print("Results saved inside outputs/")
