# Object Detection & Tracking System

A real-time object detection and tracking system built with YOLOv8 + SORT.

## Project Structure

`
task-4/
+-- main.py           <- Main application entry point
+-- sort.py           <- SORT tracking algorithm (Kalman + Hungarian)
+-- requirements.txt  <- Python dependencies
+-- README.md         <- This file
`

## Quick Start

### 1. Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run with webcam
python main.py

### 4. Run with a video file
Edit CONFIG in main.py -> change "source": 0  to  "source": "your_video.mp4"
python main.py

## Controls
  Q or ESC  - Quit
  P         - Pause/Resume
  S         - Save screenshot

## Troubleshooting
- Cannot open webcam   -> Try source: 1 or source: 2
- No detections        -> Lower confidence to 0.25
- Slow performance     -> Use yolov8n.pt (already default)
- Missing packages     -> pip install -r requirements.txt
