
# AI Smart Surveillance System

An AI-powered surveillance prototype using YOLO, OpenCV, and Streamlit to detect people, track movement, and recognize basic human behaviours in real time.

## Features

- Human and vehicle detection
- Real-time webcam detection
- Human pose estimation
- Person tracking using ByteTrack
- Basic behaviour recognition:
  - Standing
  - Sitting
  - Walking
  - Running
- Restricted-zone detection
- Live Streamlit dashboard
- CSV behaviour history

## Technologies Used

- Python
- YOLO26
- Ultralytics
- OpenCV
- PyTorch
- Streamlit
- ByteTrack

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the required YOLO model weights and place them in the project folder:
   - `yolo26n.pt`
   - `yolo26n-pose.pt`

## Run the Project

Run human behaviour detection:

```bash
python human_behaviour.py
```

Run the dashboard in a separate terminal:

```bash
streamlit run dashboard.py
```

Other scripts:

```bash
python detect_image.py
python detect_webcam.py
python pose_zone_webcam.py
```

## Disclaimer

This is an experimental AI surveillance prototype. Behaviour classifications are approximate and may be inaccurate. It should not be used as the sole basis for security decisions.
## Author

**abhiexy**