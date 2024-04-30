import os
import cv2
import json
from sahi.predict import get_sliced_prediction
from sahi.models.yolov8 import Yolov8DetectionModel  # Adjust based on actual model class
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def draw_keypoints_on_frame(frame, keypoints, output_dir, frame_index):
    """
    Draws keypoints on the given frame and saves the result.
    """
    height, width, _ = frame.shape
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    color_point1 = 'yellow'
    color_point2 = 'blue'
    color_line = 'green'

    for pair in keypoints:
        point1, point2 = pair  # Unpack the pair of keypoints
        ax.scatter(point1[0], point1[1], c=color_point1, s=40, zorder=2, edgecolors='black')  # Add edge for visibility
        ax.scatter(point2[0], point2[1], c=color_point2, s=40, zorder=2, edgecolors='black')  # Add edge for visibility
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color=color_line, linewidth=2, zorder=1)

    ax.axis('off')
    plt.savefig(os.path.join(output_dir, f'plot_keypoints_{frame_index:04d}.jpg'), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_frame_with_keypoints(frame, model, output_dir, frame_index):
    """
    Processes a single frame to detect keypoints using the model,
    then draws these keypoints on the frame and saves it.
    """
    # Convert frame to RGB as SAHI might expect RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = get_sliced_prediction(
        frame_rgb,  # Pass the frame directly
        model,
        slice_height = 320,
        slice_width = 320,
        overlap_height_ratio = 0.4,
        overlap_width_ratio = 0.4
    )

    # Extract keypoints from the first result object
    if result.object_prediction_list:
        keypoints = []  # Initialize an empty list to store keypoints
        for object_prediction in result.object_prediction_list:
            # Check if the object_prediction has keypoints attribute
            if hasattr(object_prediction, 'keypoints') and object_prediction.keypoints is not None:
                kp_xy = object_prediction.keypoints.xy.cpu().numpy()  # Extract keypoints
                keypoints.extend(kp_xy)  # Add the keypoints to the list

        if keypoints:
            draw_keypoints_on_frame(frame, keypoints, output_dir, frame_index)


def process_video_for_pose(video_path, model, output_dir):
    """
    Processes a video file to detect poses in each frame using the given model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # No more frames to process

        process_frame_with_keypoints(frame, model, output_dir, frame_index)
        frame_index += 1

    cap.release()

# Model loading logic
model_path = '/Users/apaula/Library/CloudStorage/GoogleDrive-elysiacristata@gmail.com/My Drive/datasets/runs/pose/train2/weights/best.pt'
model = Yolov8DetectionModel(
    model_path=model_path,
    confidence_threshold=0.3,
    device="cuda:0"  # Adjust as necessary, use "cpu" if CUDA is not available
)

# Example usage
video_path = '/Users/apaula/Downloads/fishschool.mp4'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"{timestamp}_poseDetections/"

process_video_for_pose(video_path, model, output_dir)
