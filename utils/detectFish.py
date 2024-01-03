import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import os

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
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            all_boxes.append((x1 + position[0], y1 + position[1], x2 + position[0], y2 + position[1]))
            all_classes.append(detection['class'])

        # Cleanup: Delete the temporary tile file
        os.remove(temp_tile_path)

    return all_boxes, all_classes

def draw_boxes_on_frame_with_color_wheel(frame, boxes, classes, class_labels, colors, frame_index):
    # Create a subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(120, 60), gridspec_kw={'width_ratios': [3, 1]})
    # Plot the frame with bounding boxes
    ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        color = colors[cls % len(colors)]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
    ax1.axis('off')

    # Create the color wheel in the second subplot
    num_classes = len(class_labels)
    label_to_angle = {'east': 0, 'northeast': np.pi/4, 'north': np.pi/2, 'northwest': 3*np.pi/4,
                      'west': np.pi, 'southwest': 5*np.pi/4, 'south': 3*np.pi/2, 'southeast': 7*np.pi/4}
    angles = [label_to_angle[label] for label in class_labels]
    ax2 = plt.subplot(122, polar=True)
    ax2.set_theta_zero_location('E')
    ax2.set_theta_direction(1)
    ax2.set_xticks(angles)
    ax2.set_xticklabels(class_labels)
    ax2.set_yticklabels([])
    for ang, color in zip(angles, colors):
        ax2.bar(ang, 1, width=2*np.pi/num_classes, color=color, bottom=0.4)

    plt.tight_layout()
    plt.savefig(f'plot_{frame_index}.png')
    plt.close()
    print(f"Plot saved as: plot_{frame_index}.png")

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

def process_video_and_save_detections(video_path, num_samples, model_path, output_path):
    # Load the YOLO model
    model = YOLO(model_path)
    frames = sample_frames_from_video(video_path, num_samples)
    all_detections = []  # To store detections for all frames

    for i, frame in enumerate(frames):
        tiles = tile_image(frame)
        boxes, classes = detect_objects_on_tiles(tiles, model)
        all_detections.append({'boxes': boxes, 'classes': classes})

        # Optional: Draw and save the plot for each frame
        # draw_boxes_on_frame_with_color_wheel(frame, boxes, classes, class_labels, colors, i)

    # Save all detections to a file
    with open(output_path, 'w') as file:
        json.dump(all_detections, file)

    print(f"All detections saved to {output_path}")

# Example usage
model_path = '/Users/apaula/Downloads/fishschool_0102.pt'
video_path = '/Users/apaula/Downloads/fishschool_trimmed.mp4'
num_samples = 5
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

process_video_and_save_detections(video_path, num_samples, model_path, output_path)
