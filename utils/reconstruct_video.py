import cv2
import os
from glob import glob

# Input and output paths
frames_dir = r"dataset/restored_frames"
output_video_path = r"dataset/restored_video/output.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# Find all PNG frames (assuming names like frame_0000corrected.png)
frame_files = sorted(glob(os.path.join(frames_dir, "*corrected.png")))

# Debug: Show a few frames found
print(f"🧾 Found {len(frame_files)} corrected PNG frames")
if len(frame_files) > 0:
    print("First few frames:", frame_files[:5])

# Check if any frames were found
if not frame_files:
    raise FileNotFoundError(f"❌ No corrected PNG frames found in: {frames_dir}")

# Read first frame to get dimensions
first_frame = cv2.imread(frame_files[0])
if first_frame is None:
    raise ValueError(f"❌ Unable to read the first frame: {frame_files[0]}")

height, width, _ = first_frame.shape

# Use a higher fps for smooth video
fps = 7.5  # Adjust as needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write frames to video
for file in frame_files:
    frame = cv2.imread(file)
    if frame is None:
        print(f"⚠️ Skipping unreadable frame: {file}")
        continue
    out.write(frame)

out.release()
print("✅ Smooth video saved to:", output_video_path)
