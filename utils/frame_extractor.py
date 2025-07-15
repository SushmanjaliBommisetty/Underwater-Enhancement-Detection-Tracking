import cv2
import os
# Paths
DATASET_DIR = "D:/dataset"
video_path = os.path.join(DATASET_DIR, "raw_video", "input.mp4")
output_dir = os.path.join(DATASET_DIR, "frames")

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 5
count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_rate == 0:
        cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
        saved += 1
    count += 1
cap.release()
