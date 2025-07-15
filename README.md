# NIOT Coral Project

This project focuses on a comprehensive pipeline for underwater object detection and tracking, incorporating image enhancement, color restoration, and bearing angle/range calculation.

## Introduction

The "NIOT Coral Project" aims to process underwater video footage for identifying and tracking specific objects. The pipeline involves several stages:
1.  **Video Preprocessing:** Extracting frames from raw video and enhancing their quality through color restoration and image enhancement.
2.  **Object Detection:** Utilizing a YOLOv8 model trained on a custom dataset of enhanced underwater images.
3.  **Object Tracking:** Tracking detected objects across frames to understand their movement.
4.  **Spatial Analysis:** Calculating the bearing angle and range of the tracked objects.

## Project Structure

NIOT_CORAL_PROJECT/
├── bearing_angle_and_range/
│   ├── bearing_and_range.py
│   └── bearing_and_range_video/
├── dataset/
│   ├── bearing_and_range_video/
│   ├── frames/
│   ├── raw_video/
│   │   └── input.mp4
│   ├── restored_frames/
│   ├── restored_video/
│   ├── tracking_video/
│   └── bearing_and_range.mp4  (Output of bearing_and_range.py)
├── detection/
│   ├── runs/
│   │   ├── detect/
│   │   │   ├── predict/
│   │   │   └── train/
│   │   │       ├── weights/
│   │   │       │   ├── best.pt
│   │   │       │   └── last.pt
│   │   │       ├── args.yaml
│   │   │       ├── BoxF1_curve.png
│   │   │       ├── BoxP_curve.png
│   │   │       ├── BoxR_curve.png
│   │   │       ├── confusion_matrix_normalized.png
│   │   │       ├── confusion_matrix.png
│   │   │       ├── labels_correlogram.jpg
│   │   │       ├── labels.jpg
│   │   │       ├── results.csv
│   │   │       └── results.png
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── data.yaml
│   └── README.roboflow.txt
├── image_enhancement_and_color_restoration/
│   ├── PyTorch_approach/
│   │   ├── pycache/
│   │   ├── checkpoints/
│   │   ├── results/
│   │   ├── test_img/
│   │   ├── model_var.pb
│   │   ├── model.onnx
│   │   ├── model.py
│   │   ├── PyTorchTraining.py
│   │   ├── test.py
│   │   ├── train.py
│   │   └── venv/
├── tracking/
│   ├── tracked_ids/
│   │   └── tracked_ids.csv
│   └── tracking_alg.py
└── utils/
├── frame_extractor.py
├── reconstruct_video.py
└── yolo11n.pt


## Workflow

The project follows a sequential workflow:

1.  **Input Video Placement:** The raw input video (`input.mp4`) is placed in `dataset/raw_video/`.
2.  **Frame Extraction:** `utils/frame_extractor.py` is used to extract individual frames from `input.mp4` and save them into `dataset/frames/`.
3.  **Image Enhancement and Color Restoration:**
    * Frames from `dataset/frames/` are copied to `image_enhancement_and_color_restoration/PyTorch_approach/test_img/`.
    * `image_enhancement_and_color_restoration/PyTorch_approach/test.py` is executed to apply color restoration and image enhancement. The enhanced frames are saved in `dataset/restored_frames/`.
4.  **Dataset Creation and Annotation (Roboflow):**
    * A Roboflow project is created using the images from `dataset/restored_frames/`.
    * Objects within these images are annotated.
    * The annotated dataset is downloaded in YOLOv8 format and placed into the `detection/` folder. This includes `detection/train/`, `detection/valid/`, and `detection/test/` directories along with `data.yaml`.
5.  **Object Detection (YOLOv8):**
    * Object detection training and prediction are performed using a Google Colab notebook ([https://colab.research.google.com/drive/1lcXMy3SCpNc4RJ9-Z9olrgVnQbbJ2W1c#scrollTo=YK6KK-Hi8g7j&uniqifier=1](https://colab.research.google.com/drive/1lcXMy3SCpNc4RJ9-Z9olrgVnQbbJ2W1c#scrollTo=YK6KK-Hi8g7j&uniqifier=1)).
    * The results, including `runs/detect/predict/`, `runs/detect/train/weights/best.pt`, `runs/detect/train/weights/last.pt`, and other training metrics (e.g., `BoxF1_curve.png`, `confusion_matrix.png`), are downloaded and placed in `detection/runs/`.
6.  **Video Reconstruction:** `utils/reconstruct_video.py` is used to combine the `dataset/restored_frames/` back into a video, saved as `dataset/restored_video/restored.mp4`.
7.  **Object Tracking:**
    * `tracking/tracking_alg.py` is executed to track detected objects within `dataset/restored_video/restored.mp4`.
    * The tracked video is saved to `dataset/tracking_video/tracking.mp4`.
    * The tracked object IDs and their corresponding data are saved in `tracking/tracked_ids/tracked_ids.csv`.
8.  **Bearing Angle and Range Calculation:**
    * `bearing_angle_and_range/bearing_and_range.py` is run on `dataset/tracking_video/tracking.mp4`.
    * This script calculates and overlays the bearing angle and range of the tracked objects, saving the final video to `dataset/bearing_angle_and_range_video/bearing_and_range.mp4`.

## Setup and Installation

**(Instructions for setting up the environment, dependencies, etc. would go here. This section is a placeholder as specific commands were not provided.)**

* **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd NIOT_CORAL_PROJECT
    ```
* **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
* **Install dependencies:**
    **(You would list the `pip install` commands for all necessary libraries here, e.g., `opencv-python`, `pytorch`, `ultralytics`, `roboflow`, `numpy`, etc.)**

## Usage

**(Detailed instructions on how to run each step of the pipeline. You would provide specific commands or script execution instructions here.)**

1.  **Place your input video:** `dataset/raw_video/input.mp4`
2.  **Extract frames:**
    ```bash
    python utils/frame_extractor.py
    ```
3.  **Prepare frames for enhancement:**
    Copy contents of `dataset/frames/` to `image_enhancement_and_color_restoration/PyTorch_approach/test_img/`
4.  **Perform image enhancement and color restoration:**
    ```bash
    python image_enhancement_and_color_restoration/PyTorch_approach/test.py
    ```
5.  **Download and place Roboflow dataset:**
    Ensure your YOLOv8 dataset structure is within `detection/` as described in [Project Structure](#project-structure).
6.  **Run Object Detection Training/Prediction:**
    Follow the instructions in the provided Google Colab notebook: [https://colab.research.google.com/drive/1lcXMy3SCpNc4RJ9-Z9olrgVnQbbJ2W1c#scrollTo=YK6KK-Hi8g7j&uniqifier=1](https://colab.research.google.com/drive/1lcXMy3SCpNc4RJ9-Z9olrgVnQbbJ2W1c#scrollTo=YK6KK-Hi8g7j&uniqifier=1). After execution, download `runs/detect/predict`, `best.pt`, `last.pt`, and other training artifacts to their respective locations within `detection/runs/`.
7.  **Reconstruct video from restored frames:**
    ```bash
    python utils/reconstruct_video.py
    ```
8.  **Track objects:**
    ```bash
    python tracking/tracking_alg.py
    ```
9.  **Calculate bearing angle and range:**
    ```bash
    python bearing_angle_and_range/bearing_and_range.py
    ```

## Results

* **Enhanced Frames:** `dataset/restored_frames/`
* **YOLOv8 Training Results:** `detection/runs/detect/train/` (includes performance curves, confusion matrices, weights)
* **YOLOv8 Prediction Results:** `detection/runs/detect/predict/`
* **Restored Video:** `dataset/restored_video/restored.mp4`
* **Tracked Video:** `dataset/tracking_video/tracking.mp4`
* **Tracked IDs:** `tracking/tracked_ids/tracked_ids.csv`
* **Bearing Angle & Range Video:** `dataset/bearing_angle_and_range_video/bearing_and_range.mp4`


