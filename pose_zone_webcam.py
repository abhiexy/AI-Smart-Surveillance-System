
import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# =====================================
# CONFIGURATION
# =====================================

MODEL_NAME = "yolo26n-pose.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.40
IMAGE_SIZE = 640

# Restricted zone polygon coordinates
# Change these points to adjust the zone
ZONE_POINTS = np.array([
    [350, 100],
    [630, 100],
    [630, 470],
    [350, 470]
], dtype=np.int32)

# Alert cooldown
ALERT_COOLDOWN = 3

# =====================================
# LOAD MODEL
# =====================================

print("Loading YOLO Pose model...")

model = YOLO(MODEL_NAME)

print("Pose model loaded successfully!")

# =====================================
# OPEN WEBCAM
# =====================================

camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Cannot access webcam!")
    raise SystemExit

print("Surveillance started. Press Q to exit.")

# =====================================
# STATE
# =====================================

previous_time = time.time()
last_alert_time = 0

person_count = 0
restricted_count = 0

# =====================================
# MAIN LOOP
# =====================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            break

        height, width = frame.shape[:2]

        # Run pose estimation
        results = model.predict(
            frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
        )

        result = results[0]

        person_count = 0
        restricted_count = 0

        # Draw restricted zone
        cv2.polylines(
            frame,
            [ZONE_POINTS],
            True,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "RESTRICTED ZONE",
            (ZONE_POINTS[0][0], max(25, ZONE_POINTS[0][1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # Process detected people
        if result.boxes is not None and result.keypoints is not None:

            for index, box in enumerate(result.boxes):

                person_count += 1

                # Bounding box
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                # Person centre
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Check restricted zone
                inside_zone = cv2.pointPolygonTest(
                    ZONE_POINTS,
                    (float(center_x), float(center_y)),
                    False
                ) >= 0

                # Extract keypoints
                keypoints = result.keypoints.data[index].cpu().numpy()

                # COCO keypoint indices
                # 5, 6 = shoulders
                # 11, 12 = hips
                # 13, 14 = knees

                def valid_point(point):
                    return point[2] > 0.35

                left_shoulder = keypoints[5]
                right_shoulder = keypoints[6]

                left_hip = keypoints[11]
                right_hip = keypoints[12]

                left_knee = keypoints[13]
                right_knee = keypoints[14]

                activity = "Uncertain posture"

                required = [
                    left_shoulder,
                    right_shoulder,
                    left_hip,
                    right_hip
                ]

                if all(valid_point(p) for p in required):

                    shoulder_y = (
                        left_shoulder[1] + right_shoulder[1]
                    ) / 2

                    hip_y = (
                        left_hip[1] + right_hip[1]
                    ) / 2

                    torso_length = hip_y - shoulder_y

                    if (
                        valid_point(left_knee)
                        and valid_point(right_knee)
                        and torso_length > 0
                    ):

                        knee_y = (
                            left_knee[1] + right_knee[1]
                        ) / 2

                        hip_knee_distance = abs(knee_y - hip_y)

                        if hip_knee_distance < torso_length * 0.45:
                            activity = "Sitting"
                        else:
                            activity = "Standing"

                    else:
                        activity = "Upright posture"

                # Choose display colour
                if inside_zone:
                    color = (0, 0, 255)
                    restricted_count += 1
                else:
                    color = (0, 255, 0)

                # Draw person bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Display activity
                label = f"Person | {activity}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                # Display person centre
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (255, 255, 0),
                    -1
                )

                # Restricted zone alert
                if inside_zone:

                    current_time = time.time()

                    if current_time - last_alert_time >= ALERT_COOLDOWN:

                        timestamp = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                        print(
                            f"[ALERT] Person entered restricted zone! "
                            f"Time: {timestamp}"
                        )

                        last_alert_time = current_time

        # Draw pose skeleton
        annotated_frame = result.plot(
            img=frame,
            boxes=False
        )

        # Calculate FPS
        current_time = time.time()

        fps = 1 / max(
            current_time - previous_time,
            0.001
        )

        previous_time = current_time

        # Dashboard overlay
        cv2.rectangle(
            annotated_frame,
            (10, 10),
            (270, 125),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            annotated_frame,
            "AI SMART SURVEILLANCE",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        cv2.putText(
            annotated_frame,
            f"People: {person_count}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            annotated_frame,
            f"Restricted zone: {restricted_count}",
            (20, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

        cv2.imshow(
            "AI Surveillance - Pose and Zone Detection",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("Surveillance stopped.")
