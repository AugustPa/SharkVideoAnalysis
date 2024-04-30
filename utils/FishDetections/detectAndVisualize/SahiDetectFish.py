import os
import cv2
import json
from sahi.predict import get_sliced_prediction
from sahi.models.yolov8 import Yolov8DetectionModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

def draw_boxes_on_frame(frame, boxes, class_ids, colors, output_dir, frame_index):
    # Get frame dimensions
    height, width, _ = frame.shape

    # Create a figure with a consistent aspect ratio
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))  # size in inches

    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for box, cls_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        color = colors[cls_id % len(colors)]
        direction = directions[cls_id % len(directions)]
        ax.arrow(x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2, direction[0], direction[1], head_width=10, head_length=10, fc=color, ec=color, linewidth=3)
    
    ax.axis('off')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    # Save the frame with a consistent DPI
    formatted_frame_index = str(frame_index).zfill(4)
    plt.savefig(os.path.join(output_dir, f'plot_{formatted_frame_index}.jpg'), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_frame(frame, model, output_dir, frame_index):
    # Convert frame to a format suitable for detection (saving to a temporary image file)
    temp_image_path = os.path.join(output_dir, 'temp_frame.jpg')
    cv2.imwrite(temp_image_path, frame)

    # Perform sliced detection
    result = get_sliced_prediction(
        temp_image_path,
        model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.4,
        overlap_width_ratio=0.4
    )

    # Extract boxes and class IDs from result
    boxes = []
    class_ids = []
    json_results = []
    for object_prediction in result.object_prediction_list:
        bbox = [float(coord) for coord in object_prediction.bbox.to_voc_bbox()]
        class_id = int(object_prediction.category.id)
        boxes.append(bbox)
        class_ids.append(class_id)
        json_results.append({"bbox": bbox, "class_id": class_id})

    # Save JSON results
        formatted_frame_index = str(frame_index).zfill(4)
    with open(os.path.join(output_dir, f'predictions_{formatted_frame_index}.json'), 'w') as f:
        json.dump(json_results, f, indent=4)

    # Draw boxes on frame
    draw_boxes_on_frame(frame, boxes, class_ids, colors, output_dir, frame_index)

def process_video(video_path, model_path, output_dir):
    # Load the model
    model = Yolov8DetectionModel(
        model_path=model_path,
        confidence_threshold=0.3,
        device="cuda:0"  # or "cpu"
    )

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frame is captured

        process_frame(frame, model, output_dir, frame_index)
        frame_index += 1

    cap.release()

# Colors and Directions for different classes
colors = [
    (1.0, 0.0, 0.0),       # East - Red
    (0.31, 1.0, 0.0),      # North - Bright Green
    (1.0, 0.84, 0.0),      # Northeast - Gold
    (0.0, 1.0, 0.529),     # Northwest - Turquoise
    (0.996, 0.0, 0.937),   # South - Pink
    (1.0, 0.0, 0.5),      # Southeast - Scarlet
    (0.216, 0.0, 1.0),     # Southwest - Electric Blue
    (0.0, 0.624, 1.0)      # West - Sky Blue
]

directions = [
    (28.3,0),   #'East
    (0,-28.3),  #'North'
    (20,-20),   #'Northeast'
    (-20,-20),  #'Northwest'
    (0,28.3),   #'South'
    (20,20),    #'Southeast'
    (-20,20),   #'Southwest'
    (-20,0)     #'West'
    ]
# Paths and parameters
video_path = '/Users/apaula/Downloads/fishschool_trimmed.mp4'
model_path = '/Users/apaula/Downloads/fishschool_0102.pt'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"{timestamp}_sahiDetections/"

process_video(video_path, model_path, output_dir)
