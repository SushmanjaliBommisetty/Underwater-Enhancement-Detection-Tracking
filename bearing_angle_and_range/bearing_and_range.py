import cv2
import numpy as np
import os
import csv

# --- Input/Output Paths ---
video_path = "D:/project/dataset/tracking_video/output.mp4"
output_path = "D:/project/dataset/bearing_and_range_video/bearing_and_range.mp4"
tracking_csv = "tracking/tracked_ids/tracked_ids.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# --- Load tracking data ---
# Format: {frame_id: [(x_center, y_center, track_id), ...]}
tracking_data = {}
with open(tracking_csv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        frame = int(row['frame'])
        x_center = float(row['x_center'])
        y_center = float(row['y_center'])
        track_id = int(row['track_id'])
        if frame not in tracking_data:
            tracking_data[frame] = []
        tracking_data[frame].append((x_center, y_center, track_id))

# --- Video Setup ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Expected: 1920
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Expected: 1080
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# --- Camera Center and Focal Length ---
cx = frame_width / 2
cy = frame_height / 2
fx = 1000  # approximate focal length in pixels

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get tracked objects for this frame
    tracked_objects = tracking_data.get(frame_id, [])

    for (x_center, y_center, track_id) in tracked_objects:
        # Bearing angle (left/right from center)
        dx = x_center - cx
        bearing_rad = np.arctan2(dx, fx)
        bearing_deg = np.degrees(bearing_rad)

        # Pixel 2D range from center
        dy = y_center - cy
        pixel_range = np.sqrt(dx**2 + dy**2)

        # --- Drawing ---
        cv2.arrowedLine(frame, (int(cx), int(cy)), (int(x_center), int(y_center)),
                        (0, 255, 0), 2, tipLength=0.05)
        label = f"ID {track_id} | BA: {bearing_deg:.1f}° | R: {pixel_range:.1f}px"
        cv2.putText(frame, label, (int(x_center - 100), int(y_center - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 255, 255), -1)

    out.write(frame)
    frame_id += 1

cap.release()
out.release()
print(f"[✅] Done. Saved to: {output_path}")