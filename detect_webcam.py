
import cv2
import time
from collections import Counter
from ultralytics import YOLO

MODEL_NAME = "yolo26n.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.35
IMAGE_SIZE = 640

TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

model = YOLO(MODEL_NAME)

camera = cv2.VideoCapture(CAMERA_INDEX)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Cannot access webcam!")
    exit()

previous_time = time.time()

try:
    while True:

        success, frame = camera.read()

        if not success:
            break

        results = model.predict(
            frame,
            classes=TARGET_CLASSES,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        counts = Counter()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                counts[CLASS_NAMES[class_id]] += 1

        total_objects = sum(counts.values())

        current_time = time.time()
        fps = 1 / max(current_time - previous_time, 0.001)
        previous_time = current_time

        # Dashboard overlay
        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (260, 230),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            annotated_frame,
            "AI SURVEILLANCE",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Total Objects: {total_objects}",
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        y = 128

        for name in CLASS_NAMES.values():
            cv2.putText(
                annotated_frame,
                f"{name}: {counts.get(name, 0)}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y += 19

        cv2.imshow(
            "AI Smart Surveillance",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Surveillance stopped.")
