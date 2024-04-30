import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import os
import datetime

def tile_image(image, tile_size=640, overlap=0.2):
    """
    Tile the image into smaller overlapping tiles.
    Returns a list of tiles and their positions in the original image.
    """
    tiles = []
    step = int(tile_size * (1 - overlap))
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            # Adjust the tile size if we are at the edge of the image
            adjusted_tile_size_y = min(tile_size, image.shape[0] - y)
            adjusted_tile_size_x = min(tile_size, image.shape[1] - x)
            tile = image[y:y + adjusted_tile_size_y, x:x + adjusted_tile_size_x]
            tiles.append((tile, (x, y)))
    return tiles

def detect_objects_on_tiles(tiles, model):
    all_boxes = []
    all_scores = []
    all_classes = []
    for tile, position in tiles:
        # Save the tile temporarily (required for your model's input format)
        temp_tile_path = 'temp_tile.jpg'
        cv2.imwrite(temp_tile_path, tile)

        # Run detection on the tile
        results = model(temp_tile_path)
        json_results = json.loads(results[0].tojson())

        # Adjust box coordinates and extract class information based on the tile position
        for detection in json_results:
            box = detection['box']
            score = detection['confidence']  # Replace with your model's confidence score field
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            all_boxes.append([x1 + position[0], y1 + position[1], x2 + position[0], y2 + position[1]])
            all_scores.append(score)
            all_classes.append(detection['class'])

        # Cleanup: Delete the temporary tile file
        os.remove(temp_tile_path)

    # Convert boxes for NMS
    boxes_for_nms = [[x, y, x2-x1, y2-y1] for [x, y, x2, y2] in all_boxes]

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, all_scores, score_threshold=0.01, nms_threshold=0.999)

    # Filter out boxes using indices from NMS
    unique_boxes = []
    unique_classes = []
    for index in indices:
        # Check if index is a scalar (i.e., a single number) or an array
        if np.isscalar(index):
            i = index
        else:
            i = index[0]  # Extract the actual index value

        # Append the unique boxes and classes
        unique_boxes.append(all_boxes[i])
        unique_classes.append(all_classes[i])

    return unique_boxes, unique_classes, tiles

def draw_boxes_on_frame(frame, boxes, classes, class_labels, colors, frame_index, output_dir):
    fig, ax = plt.subplots(figsize=(64, 36))  # Adjust for 4K resolution
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        color = colors[cls % len(colors)]
        # draw the center point instead of th e box
        rect = patches.Circle((x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2), radius=5, linewidth=8, edgecolor=color, facecolor='none')
        #rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')
    plt.tight_layout()
    formatted_frame_index = str(frame_index).zfill(4)
    plt.savefig(os.path.join(output_dir, f'plot_{formatted_frame_index}.png'), dpi=75)
    plt.close()

def draw_boxes_on_tile(tile, position, boxes, classes, class_labels, colors):
    # Create a copy of the tile to draw on
    tile_with_boxes = tile.copy()

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        # Adjust box coordinates based on the tile position
        x1, x2 = x1 - position[0], x2 - position[0]
        y1, y2 = y1 - position[1], y2 - position[1]

        # Draw the bounding box on the tile
        color = colors[cls % len(colors)]
        cv2.rectangle(tile_with_boxes, (x1, y1), (x2, y2), color, 2)
        label = class_labels[cls]
        cv2.putText(tile_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return tile_with_boxes

def sample_frames_from_video(video_path, num_samples):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_frames = []
    for i in range(num_samples):
        frame_id = (frame_count // num_samples) * i
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
    cap.release()
    return sampled_frames

def process_video_and_save_detections(video_path, model_path, output_folder, start_frame=0, end_frame=None):
    model = YOLO(model_path)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_folder, current_time)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If end_frame is not specified or is beyond the total frame count, set it to the last frame
    if end_frame is None or end_frame > frame_count:
        end_frame = frame_count

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detections_file = os.path.join(output_dir, 'detections.json')
    with open(detections_file, 'a') as file:
        tile_index = 0
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            tiles = tile_image(frame)
            boxes, classes, tiles_with_positions = detect_objects_on_tiles(tiles, model)

            # Process each tile
            for (tile, position) in tiles_with_positions:
                tile_boxes = [box for box, (x, y, x2, y2) in zip(boxes, boxes) if position[0] <= x < position[0] + tile.shape[1] and position[1] <= y < position[1] + tile.shape[0]]
                tile_classes = [cls for cls, box in zip(classes, boxes) if position[0] <= box[0] < position[0] + tile.shape[1] and position[1] <= box[1] < position[1] + tile.shape[0]]
                
                tile_with_boxes = draw_boxes_on_tile(tile, position, tile_boxes, tile_classes, class_labels, colors)
                tile_file_path = os.path.join(output_dir, f'tile_{tile_index}.jpg')
                cv2.imwrite(tile_file_path, tile_with_boxes)

                # Write tile information and detections
                tile_data = {'tile_index': tile_index, 'position': position, 'boxes': tile_boxes, 'classes': tile_classes}
                json.dump(tile_data, file)
                file.write("\n")

                tile_index += 1

            # Draw and save the plot for each frame
            draw_boxes_on_frame(frame, boxes, classes, class_labels, colors, i+1, output_dir)

    cap.release()
    print(f"All detections saved to {detections_file}")


# Example usage
model_path = '/Users/apaula/Downloads/fishschool_0102.pt'
video_path = '/Users/apaula/Downloads/fishschool_trimmed.mp4'
output_folder = '/Users/apaula/Downloads/'
num_samples = 100
output_path = 'detections.json'
class_labels = ['east', 'north', 'northeast', 'northwest', 'south', 'southeast', 'southwest', 'west']
colors = [
    (1.0, 0.0, 0.0),       # East - Red
    (0.31, 1.0, 0.0),      # North - Bright Green
    (1.0, 0.84, 0.0),      # Northeast - Gold
    (0.0, 1.0, 0.529),     # Northwest - Turquoise
    (0.996, 0.0, 0.937),   # South - Pink
    (1.0, 0.0, 0.09),      # Southeast - Scarlet
    (0.216, 0.0, 1.0),     # Southwest - Electric Blue
    (0.0, 0.624, 1.0)      # West - Sky Blue
]
process_video_and_save_detections(video_path, model_path, output_folder,start_frame=4790, end_frame=4791)
