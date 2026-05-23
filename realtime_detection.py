import cv2
import time
import csv
import argparse
import os
import winsound
from datetime import datetime
from ultralytics import YOLO

# ─── Configuration & Arguments ───────────────────────────────────────
parser = argparse.ArgumentParser(description="Real-Time Construction Safety Monitor")
parser.add_argument("--weights", type=str, default="best.pt", help="Path to your YOLOv8 weights")
parser.add_argument("--source", type=str, default="0", help="Webcam index (0) or path to video file (.mp4)")
args = parser.parse_args()

# ─── Load Model ──────────────────────────────────────────────────────
print(f"Loading YOLO model from: {args.weights}")
try:
    model = YOLO(args.weights)
    
    # UPGRADE: Override the default dataset names to look better on screen!
    # Original was {0: 'head', 1: 'helmet', 2: 'person'}
    model.model.names = {0: "No Helmet", 1: "Helmet", 2: "Person"}
    
except Exception as e:
    print(f" Error loading model: {e}")
    print("Make sure 'best.pt' is in the same folder as this script!")
    exit()

# ─── Initialize Video Stream ─────────────────────────────────────────
source = int(args.source) if args.source.isdigit() else args.source

# Force Windows DirectShow to prevent the camera from freezing
if isinstance(source, int):
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f" Error: Could not open video source '{args.source}'")
    exit()

# ─── Setup OpenCV Window & Confidence Slider ─────────────────────────
window_name = "Task 4: Real-Time Safety Monitor"
cv2.namedWindow(window_name)

# Dummy callback for the trackbar
def on_trackbar(val):
    pass

# UPGRADE: Set default slider to 85% to filter out false "helmet" detections on bare hair
cv2.createTrackbar("Confidence (%)", window_name, 85, 100, on_trackbar)

# ─── Setup Bonus Feature: Violation Logging ──────────────────────────
log_filename = "safety_violations_log.csv"
if not os.path.exists(log_filename):
    with open(log_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Violation Type", "Confidence"])

prev_time = time.time()
cooldown_time = 0  # Prevents logging the same violation 30 times a second

print("\n✅ System Ready! Press 'q' in the video window to quit.")

# ─── Main Inference Loop ─────────────────────────────────────────────
while True:
    # Read the frame ONLY ONCE per loop
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot fetch frame.")
        break

    # Resize to a smaller resolution (e.g., 480p) to speed up CPU processing
    frame = cv2.resize(frame, (640, 480))

    # 1. Read slider value and convert to decimal (e.g., 85 -> 0.85)
    slider_val = cv2.getTrackbarPos("Confidence (%)", window_name)
    conf_thresh = max(0.05, slider_val / 100.0) 

    # 2. Run YOLO inference on the frame
    results = model.predict(source=frame, conf=conf_thresh, verbose=False)
    
    # 3. Get the annotated frame
    annotated_frame = results[0].plot()
    
    # 4. Check for safety violations
    violation_detected = False
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # UPGRADE: Class 0 is the missing helmet class!
        if cls_id == 0:  
            violation_detected = True
            
            # BONUS ACTION: Log to CSV and play audio with a 2-second cooldown
            current_time = time.time()
            if current_time - cooldown_time > 2.0:
                
                # 1. PLAY THE AUDIO ALERT (Asynchronous background sound)
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
                
                # 2. LOG TO CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, "No Helmet", f"{conf:.2f}"])
                
                print(f" ALERT: Worker without helmet logged at {timestamp} (Confidence: {conf:.2f})")
                
                # 3. Reset the cooldown timer
                cooldown_time = current_time

    # 5. BONUS ACTION: Visual Alert
    if violation_detected:
        # Draw a thick red warning border around the whole screen
        h, w = annotated_frame.shape[:2]
        cv2.rectangle(annotated_frame, (0, 0), (w, h), (0, 0, 255), 15)
        # Add warning text
        cv2.putText(annotated_frame, " WARNING: NO HELMET DETECTED", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # 6. Calculate & Display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 7. Show the final image
    cv2.imshow(window_name, annotated_frame)

    # 8. Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── Clean Up ────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()