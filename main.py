"""
Object Detection and Tracking System
=====================================
Uses YOLOv8 (ultralytics) for detection and SORT for tracking.

Controls:
  q  — quit
  p  — pause / resume
  s  — save a screenshot
"""

import sys
import time
import datetime
import cv2
import numpy as np

# ── Try to import ultralytics (YOLOv8) ──────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] 'ultralytics' is not installed.")
    print("        Run:  pip install ultralytics")
    sys.exit(1)

# ── Try to import filterpy (required by sort.py) ─────────────────────────────
try:
    from sort import Sort
except ImportError as e:
    print(f"[ERROR] Could not import SORT tracker: {e}")
    print("        Make sure sort.py is in the same folder and filterpy is installed.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (edit these as needed)
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # --- Input ---
    "source": 0,                  # 0 = default webcam | "path/to/video.mp4"

    # --- YOLO ---
    "yolo_model": "yolov8n.pt",   # nano model downloads automatically on first run
    "confidence": 0.40,           # minimum detection confidence (0–1)
    "nms_iou": 0.45,              # Non-Maximum Suppression IoU threshold

    # --- SORT tracker ---
    "sort_max_age": 10,           # frames to keep a lost track alive
    "sort_min_hits": 2,           # detections before a track is shown
    "sort_iou_threshold": 0.30,

    # --- Display ---
    "window_title": "Object Detection & Tracking  |  Press Q to quit",
    "show_fps": True,
    "box_thickness": 2,
    "font_scale": 0.6,
}

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_color(track_id: int):
    """Generate a vivid, deterministic BGR color from a track ID."""
    np.random.seed(int(track_id) * 17 + 3)
    hue = int(np.random.randint(0, 180))
    color_hsv = np.array([[[hue, 230, 220]]], dtype=np.uint8)
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))


def draw_label_box(frame, text: str, x1: int, y1: int, color, font_scale=0.6, thickness=2):
    """Draw a filled label rectangle above the bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    rect_y1 = max(y1 - th - 2 * pad, 0)
    rect_y2 = y1
    rect_x2 = x1 + tw + 2 * pad

    cv2.rectangle(frame, (x1, rect_y1), (rect_x2, rect_y2), color, -1)
    cv2.putText(frame, text, (x1 + pad, y1 - pad),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_hud(frame, n_objects: int, fps: float, paused: bool):
    """Draw a semi-transparent HUD overlay at the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top bar background
    cv2.rectangle(overlay, (0, 0), (w, 50), (15, 15, 15), -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Object count
    count_text = f"Objects detected: {n_objects}"
    cv2.putText(frame, count_text, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 230, 50), 2, cv2.LINE_AA)

    # FPS
    if CONFIG["show_fps"]:
        fps_text = f"FPS: {fps:.1f}"
        (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, fps_text, (w - fw - 12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 200, 255), 2, cv2.LINE_AA)

    # Paused indicator
    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 120, 255), 3, cv2.LINE_AA)

    # Hotkey reminder at the bottom
    hint = "Q: Quit   P: Pause   S: Screenshot"
    cv2.putText(frame, hint, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Object Detection & Tracking System")
    print("  YOLOv8 + SORT")
    print("=" * 60)

    # ── Load YOLO model ───────────────────────────────────────────────────────
    print(f"\n[INFO] Loading YOLO model: {CONFIG['yolo_model']} …")
    try:
        model = YOLO(CONFIG["yolo_model"])
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        sys.exit(1)

    # ── Open video source ─────────────────────────────────────────────────────
    source = CONFIG["source"]
    print(f"[INFO] Opening source: {'webcam (index 0)' if source == 0 else source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        if source == 0:
            print("        • Make sure a webcam is connected.")
            print("        • Try changing 'source' to 1 or 2 for other cameras.")
        else:
            print(f"        • Check that the file exists: {source}")
        sys.exit(1)

    # Try to set a comfortable resolution for webcam
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Initialise SORT tracker ───────────────────────────────────────────────
    tracker = Sort(
        max_age=CONFIG["sort_max_age"],
        min_hits=CONFIG["sort_min_hits"],
        iou_threshold=CONFIG["sort_iou_threshold"],
    )

    # ── State variables ───────────────────────────────────────────────────────
    paused = False
    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    # Map track_id → class_name (best-effort, assigned at first match)
    id_to_class: dict[int, str] = {}
    id_to_conf:  dict[int, float] = {}

    print("\n[INFO] Starting. Press Q to quit, P to pause, S to screenshot.\n")
    cv2.namedWindow(CONFIG["window_title"], cv2.WINDOW_NORMAL)

    while True:
        # ── Handle pause ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # Q or ESC → quit
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"screenshot_{ts}.jpg"
            cv2.imwrite(fname, frame if 'frame' in dir() else np.zeros((480, 640, 3), np.uint8))
            print(f"[INFO] Screenshot saved: {fname}")

        if paused:
            if 'frame' in dir():
                draw_hud(frame, n_objects=0, fps=fps, paused=True)
                cv2.imshow(CONFIG["window_title"], frame)
            continue

        # ── Read frame ────────────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):   # end of video file → loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[WARNING] Failed to read frame from webcam. Retrying…")
            time.sleep(0.05)
            continue

        frame_count += 1

        # ── YOLO inference ────────────────────────────────────────────────────
        results = model.predict(
            source=frame,
            conf=CONFIG["confidence"],
            iou=CONFIG["nms_iou"],
            verbose=False,
        )

        # ── Parse detections [x1,y1,x2,y2,score,class_id] ────────────────────
        detections = np.empty((0, 6))
        for r in results:
            if r.boxes is None:
                continue
            boxes  = r.boxes.xyxy.cpu().numpy()    # (N,4)
            scores = r.boxes.conf.cpu().numpy()    # (N,)
            cls_ids = r.boxes.cls.cpu().numpy()    # (N,)
            if len(boxes):
                det_block = np.column_stack([boxes, scores, cls_ids])
                detections = np.vstack([detections, det_block])

        # ── SORT needs [x1,y1,x2,y2,score] ───────────────────────────────────
        sort_input = detections[:, :5] if len(detections) else np.empty((0, 5))
        tracked = tracker.update(sort_input)  # → (M, 5): [x1,y1,x2,y2,track_id]

        # ── Map track_id to class / confidence ───────────────────────────────
        # For each active track, find the detection with highest IoU and read its class
        for trk in tracked:
            x1t, y1t, x2t, y2t, tid = map(int, trk[:5])
            best_iou, best_cls, best_conf = 0.0, -1, 0.0
            for det in detections:
                x1d, y1d, x2d, y2d, score, cls_id = det
                inter_x1 = max(x1t, x1d); inter_y1 = max(y1t, y1d)
                inter_x2 = min(x2t, x2d); inter_y2 = min(y2t, y2d)
                iw = max(0, inter_x2 - inter_x1)
                ih = max(0, inter_y2 - inter_y1)
                inter = iw * ih
                union = ((x2t-x1t)*(y2t-y1t) + (x2d-x1d)*(y2d-y1d) - inter)
                iou_val = inter / union if union > 0 else 0.0
                if iou_val > best_iou:
                    best_iou, best_cls, best_conf = iou_val, int(cls_id), float(score)
            if best_cls >= 0:
                id_to_class[tid] = COCO_CLASSES[best_cls] if best_cls < len(COCO_CLASSES) else f"cls{best_cls}"
                id_to_conf[tid]  = best_conf

        # ── Draw tracked bounding boxes ───────────────────────────────────────
        for trk in tracked:
            x1, y1, x2, y2, tid = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[4])
            color = generate_color(tid)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, CONFIG["box_thickness"])

            # Corner accents for a premium look
            clen = 14
            ct = CONFIG["box_thickness"] + 1
            for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                      (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(frame, (cx, cy), (cx + dx * clen, cy), color, ct)
                cv2.line(frame, (cx, cy), (cx, cy + dy * clen), color, ct)

            # Label: "ClassName  ID:N  conf%"
            cls_name = id_to_class.get(tid, "object")
            conf_val = id_to_conf.get(tid, 0.0)
            label = f"{cls_name}  ID:{tid}  {conf_val:.0%}"
            draw_label_box(frame, label, x1, y1, color, CONFIG["font_scale"], CONFIG["box_thickness"])

        # ── HUD ───────────────────────────────────────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        draw_hud(frame, n_objects=len(tracked), fps=fps, paused=False)

        # ── Show frame ────────────────────────────────────────────────────────
        cv2.imshow(CONFIG["window_title"], frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Session ended. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
