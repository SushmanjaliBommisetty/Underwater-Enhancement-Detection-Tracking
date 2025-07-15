from ultralytics import YOLO
import cv2
import yaml
import os
import csv

# Load class names from detection/data.yaml
with open('detection/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
class_names = data['names']

# Load your trained YOLO model weights
model = YOLO("D:/project/detection/runs/detect/train/weights/best.pt")  # Update path as needed

# Load video
video_path = "dataset/restored_video/output.mp4"
cap = cv2.VideoCapture(video_path)

# Set desired output properties
output_width = 1920
output_height = 1080
output_fps = 7.5

# Output path
output_path = "dataset/tracking_video/output.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))

# Prepare to save tracking IDs
tracked_id_dir = "tracking/tracked_ids"
os.makedirs(tracked_id_dir, exist_ok=True)
csv_path = os.path.join(tracked_id_dir, "tracked_ids.csv")
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'track_id', 'x_center', 'y_center', 'width', 'height', 'class_id', 'confidence'])

frame_num = 0
ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        # Detect and track objects
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()
        # Resize output to 1920x1080
        frame_ = cv2.resize(frame_, (output_width, output_height))
        out.write(frame_)
        cv2.imshow('frame', frame_)

        # Save tracking IDs and positions
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id, cls, conf in zip(boxes.xywh, boxes.id, boxes.cls, boxes.conf):
                x_center, y_center, w, h = box.tolist()
                csv_writer.writerow([
                    frame_num,
                    int(track_id),
                    x_center,
                    y_center,
                    w,
                    h,
                    int(cls),
                    float(conf)
                ])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_num += 1

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"[✅] Tracking IDs saved to {csv_path}")