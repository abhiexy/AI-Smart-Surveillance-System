
import cv2
import time
import math
import csv
import os
import json
from collections import deque, Counter
from ultralytics import YOLO

# Load model
model = YOLO("yolo26n-pose.pt")

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Webcam not found!")
    raise SystemExit

CONF = 0.35
POSE_CONF = 0.20
INFERENCE_INTERVAL = 0.20

os.makedirs("outputs", exist_ok=True)

LOG_FILE = "outputs/behaviour_history.csv"
FRAME_FILE = "outputs/live_frame.jpg"
STATUS_FILE = "outputs/live_status.json"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp", "Person ID",
            "Behaviour", "Confidence"
        ])

tracks = {}
previous_behaviours = {}


def center(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)

    if mag == 0:
        return 180

    return math.degrees(
        math.acos(max(-1, min(1, dot / mag)))
    )


def posture(k):
    ls, rs = k[5], k[6]
    lh, rh = k[11], k[12]
    lk, rk = k[13], k[14]
    la, ra = k[15], k[16]

    if min(ls[2], rs[2], lh[2], rh[2]) < POSE_CONF:
        return "Unknown"

    knee_angles = []

    if min(lh[2], lk[2], la[2]) >= POSE_CONF:
        knee_angles.append(angle(lh, lk, la))

    if min(rh[2], rk[2], ra[2]) >= POSE_CONF:
        knee_angles.append(angle(rh, rk, ra))

    if knee_angles and min(knee_angles) < 135:
        return "Sitting"

    return "Standing"


def movement(track, c, height):
    track["history"].append((time.time(), c))

    if len(track["history"]) < 6:
        return "Standing"

    old_time, old_center = track["history"][0]
    elapsed = time.time() - old_time

    if elapsed <= 0:
        return "Standing"

    distance = math.hypot(
        c[0] - old_center[0],
        c[1] - old_center[1]
    )

    speed = distance / max(height, 1) / elapsed

    if speed > 1.20:
        return "Running"

    if speed > 0.35:
        return "Walking"

    return "Standing"


def log_event(track_id, behaviour, confidence):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            track_id,
            behaviour,
            round(float(confidence), 3)
        ])

    print(
        f"[EVENT] {timestamp} | "
        f"Person {track_id} | {behaviour}"
    )



def publish_status(people, running=True):
    data = {
        "running": running,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "people": people
    }

    temp_file = STATUS_FILE + ".tmp"

    try:
        with open(temp_file, "w") as file:
            json.dump(data, file)

        # Retry if Windows temporarily locks the file
        for attempt in range(5):
            try:
                os.replace(temp_file, STATUS_FILE)
                return
            except PermissionError:
                time.sleep(0.1)

        # Fallback: write directly
        with open(STATUS_FILE, "w") as file:
            json.dump(data, file)

    except (PermissionError, OSError) as error:
        print("Status update temporarily failed:", error)



print("Live Behaviour Detection Started")
print("Press Q or ESC to exit")

last_publish = 0
last_inference = 0
cached_results = None
current_people = []

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        now = time.time()

        # Run YOLO only once every 0.20 seconds
        if now - last_inference >= INFERENCE_INTERVAL:
            cached_results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=CONF,
                imgsz=416,
                verbose=False
            )

            last_inference = now

            current_people = []

            for result in cached_results:
                if (
                    result.keypoints is None
                    or result.boxes is None
                    or result.boxes.id is None
                ):
                    continue

                keypoints = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()

                for k, box, track_id, confidence in zip(
                    keypoints, boxes, ids, confidences
                ):
                    x1, y1, x2, y2 = map(int, box)
                    height = max(y2 - y1, 1)
                    c = ((x1 + x2) // 2, (y1 + y2) // 2)

                    if track_id not in tracks:
                        tracks[track_id] = {
                            "history": deque(maxlen=10),
                            "posture_history": deque(maxlen=12)
                        }

                    track = tracks[track_id]
                    p = posture(k)

                    if p != "Unknown":
                        track["posture_history"].append(p)

                    history = track["posture_history"]

                    if history:
                        stable_posture = Counter(
                            history
                        ).most_common(1)[0][0]
                    else:
                        stable_posture = "Unknown"

                    if stable_posture == "Unknown":
                        behaviour = "Unknown"
                    else:
                        movement_result = movement(
                            track, c, height
                        )

                        if movement_result in ("Walking", "Running"):
                            behaviour = movement_result
                        else:
                            behaviour = stable_posture

                    if behaviour != "Unknown":
                        previous = previous_behaviours.get(track_id)

                        if behaviour != previous:
                            log_event(
                                track_id,
                                behaviour,
                                confidence
                            )
                            previous_behaviours[track_id] = behaviour

                    colors = {
                        "Standing": (0, 255, 0),
                        "Sitting": (255, 150, 0),
                        "Walking": (0, 255, 255),
                        "Running": (0, 0, 255),
                        "Unknown": (150, 150, 150)
                    }

                    color = colors.get(
                        behaviour, (255, 255, 255)
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    label = f"Person {track_id}: {behaviour}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    current_people.append({
                        "id": track_id,
                        "behaviour": behaviour,
                        "confidence": round(float(confidence), 3)
                    })

        # Publish frame and status once per second
        now = time.time()

        if now - last_publish >= 1.0:
            resized_frame = cv2.resize(
                frame, (480, 360)
            )

            cv2.imwrite(
                FRAME_FILE,
                resized_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )

            publish_status(current_people)
            last_publish = now

        cv2.imshow(
            "Live Human Behaviour Analysis",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

finally:
    publish_status([], running=False)
    cap.release()
    cv2.destroyAllWindows()
