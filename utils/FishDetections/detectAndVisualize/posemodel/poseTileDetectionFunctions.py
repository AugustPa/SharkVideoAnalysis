import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ultralytics import YOLO 

# Load the YOLO model
model = YOLO('/Users/apaula/Library/CloudStorage/GoogleDrive-elysiacristata@gmail.com/My Drive/datasets/runs/pose/train9/weights/best.pt')

#this function takes a frame and tiles it into smaller images
def tile_frame(frame, tile_size=320, overlap=0.4):
    stride = int(tile_size * (1 - overlap))
    tiles = []
    tile_positions = []  # To track the original position of each tile in the frame
    
    for y in range(0, frame.shape[0] - tile_size + 1, stride):
        for x in range(0, frame.shape[1] - tile_size + 1, stride):
            tile = frame[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            tile_positions.append((x, y))
            
    return tiles, tile_positions

#this function runs the model on each tile and returns the results
def detect_pose_in_tile(tile, model, confidence_threshold=0.3, distance_range=(25, 100)):
    results = model(tile)
    detections = []

    # Accessing the first (and presumably only) Results object
    results_object = results[0]

    # Extracting bounding boxes and their confidence scores from 'data' attribute
    if hasattr(results_object, 'boxes') and results_object.boxes is not None:
        for i, box_data in enumerate(results_object.boxes.data):
            conf = box_data[4].item()  # Convert tensor to Python scalar
            if conf > confidence_threshold:
                detection = {
                    'bbox': box_data[:4].cpu().numpy(),  # Convert bbox tensor to numpy array
                    'confidence': conf,
                    'class': box_data[5].item()  # Class of the detection
                }

                # Add keypoints if available
                if hasattr(results_object, 'keypoints') and results_object.keypoints is not None and len(results_object.keypoints.data) > i:
                    kp_data = results_object.keypoints.data[i]
                    points = kp_data[:, :2].cpu().numpy()  # Keypoint coordinates

                    # Calculate distances between pairs of keypoints and apply filtering
                    if len(points) > 1:  # Ensure there are at least two points to form a pair
                        distances = np.linalg.norm(points[0] - points[1], axis=0)
                        if distance_range[0] <= distances <= distance_range[1]:
                            detection['keypoints'] = {
                                'points': points,
                                'confidence': kp_data[:, 2].cpu().numpy()  # Keypoint confidences
                            }

                detections.append(detection)

    return detections

#this function is to visualize the detections on a single tile
def draw_detections_and_save(tile, detections, save_dir, tile_index, draw_boxes=True):
    """
    Draw keypoints, lines between them with strong visibility, and optionally bounding boxes with reduced alpha on a tile and save the result to disk.
    
    Parameters:
    - tile: The image tile to draw detections on.
    - detections: A list of detections where each detection is a dictionary
      containing 'bbox' (bounding box as [xmin, ymin, xmax, ymax]), 'confidence',
      and optionally 'keypoints' with 'points' and 'confidence'.
    - save_dir: Directory to save the output images.
    - tile_index: Index of the tile to create a unique filename.
    - draw_boxes: Boolean to decide if bounding boxes should also be drawn.
    """
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a copy of the tile to draw bounding boxes with reduced opacity
    if draw_boxes:
        overlay_boxes = tile.copy()

    # Create a copy of the tile to draw keypoints and lines with stronger opacity
    overlay_keypoints = tile.copy()

    # Draw detections on the tile
    for detection in detections:
        if draw_boxes:
            bbox = detection['bbox']
            cv2.rectangle(overlay_boxes, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            confidence = detection.get('confidence', 0)
            cv2.putText(overlay_boxes, f'{confidence:.2f}', (int(bbox[0]), int(bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2)

        if 'keypoints' in detection:
            keypoints = detection['keypoints']['points']
            # Assuming keypoints are paired
            for i in range(0, len(keypoints), 2):
                point1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                point2 = (int(keypoints[i+1][0]), int(keypoints[i+1][1]))
                cv2.circle(overlay_keypoints, point1, 5, (255, 0, 0), -1)  # Draw keypoints in red
                cv2.circle(overlay_keypoints, point2, 5, (0, 0, 255), -1)  # Draw keypoints in blue
                cv2.line(overlay_keypoints, point1, point2, (0, 255, 255), 2)  # Strong yellow line

    # Blend overlays with the original image
    if draw_boxes:
        cv2.addWeighted(overlay_boxes, 0.2, tile, 0.8, 0, tile)  # Bounding boxes with reduced opacity
    cv2.addWeighted(overlay_keypoints, 0.8, tile, 0.2, 0, tile)  # Keypoints and lines with stronger opacity

    # Save the tile with detections drawn on it
    save_path = os.path.join(save_dir, f'tile_{tile_index}.jpg')
    cv2.imwrite(save_path, tile)
    
def process_tiles_and_save_detections(tiles, model, save_dir='detected_tiles'):
    """
    Process each tile for pose detection, draw the detections, and save the results.
    
    Parameters:
    - tiles: List of image tiles to process.
    - model: The loaded pose detection model.
    - save_dir: Directory to save the output images with detections.
    """
    for i, tile in enumerate(tiles):
        # Assuming your detection function returns a list of detections for the tile
        detections = detect_pose_in_tile(tile, model)
        draw_detections_and_save(tile, detections, save_dir, i)
import numpy as np

def aggregate_detections(tiled_detections, tile_positions, tile_size=320):
    full_frame_detections = []
    for tile_detections, (x_offset, y_offset) in zip(tiled_detections, tile_positions):
        for detection in tile_detections:
            bbox = detection['bbox']
            adjusted_bbox = (
                bbox[0] + x_offset,
                bbox[1] + y_offset,
                bbox[2] + x_offset,
                bbox[3] + y_offset
            )
            if 'keypoints' in detection:
                keypoints = detection['keypoints']
                adjusted_points = keypoints['points'] + np.array([x_offset, y_offset])
                adjusted_keypoints = {
                    'points': adjusted_points,
                    'confidence': keypoints['confidence']
                }
                detection['keypoints'] = adjusted_keypoints
            detection['bbox'] = adjusted_bbox
            full_frame_detections.append(detection)
    return full_frame_detections

def keypoint_similarity(kp1, kp2):
    if kp1.shape != kp2.shape:
        raise ValueError("Keypoint sets must have the same size and dimension.")
    distances = np.linalg.norm(kp1 - kp2, axis=1)
    average_distance = np.mean(distances)
    return average_distance

def remove_duplicates_by_keypoints(detections, similarity_threshold=20):
    n = len(detections)
    if n == 0:
        return []
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    removed = [False] * n
    for i in range(n):
        if not removed[i]:
            keep.append(detections[i])
            for j in range(i+1, n):
                if not removed[j] and 'keypoints' in detections[i] and 'keypoints' in detections[j]:
                    kp_sim = keypoint_similarity(detections[i]['keypoints']['points'], detections[j]['keypoints']['points'])
                    if kp_sim < similarity_threshold:
                        removed[j] = True
    return keep
# Example function that processes a video frame

def process_frame(frame, model, tile_size, overlap, confidence_threshold, distance_range):
    tiles, tile_positions = tile_frame(frame, tile_size, overlap)
    tiled_detections = [detect_pose_in_tile(tile, model, confidence_threshold, distance_range) for tile in tiles]
    aggregated_detections = aggregate_detections(tiled_detections, tile_positions, tile_size)
    final_detections = remove_duplicates_by_keypoints(aggregated_detections, similarity_threshold=20)
    return final_detections
def visualize_detections(frame, detections):
    """
    Visualize detections with bounding boxes and keypoints on the frame.

    Args:
    frame (numpy.ndarray): The original image frame.
    detections (list of dicts): List of detections, each detection is a dict with 'bbox' and 'keypoints'.
    """
    for detection in detections:
        # Draw bounding box
        bbox = detection['bbox']
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        # Draw keypoints
        if 'keypoints' in detection and 'points' in detection['keypoints']:
            for point in detection['keypoints']['points']:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    

    # Optionally, save the image
    cv2.imwrite('allDetectionsFrame.jpg', frame)
# Main execution logic for testing (modify as needed)
if __name__ == '__main__':
    # Example call to process tiles and save detections
    # Load the image
    image_path = '/Users/apaula/src/SharkVideoAnalysis/20240219_184319_poseDetections/temp_frame.jpg'
    frame = cv2.imread(image_path)
    tiles, tile_positions = tile_frame(frame)  # Unpack both tiles and their positions
    model = YOLO('/Users/apaula/Library/CloudStorage/GoogleDrive-elysiacristata@gmail.com/My Drive/datasets/runs/pose/train9/weights/best.pt')  # Load your model
    detections=process_frame(frame, model, 320, 0.4, 0.3, (25, 100))  # Process the frame and get detections
    visualize_detections(frame, detections)  # Visualize the detections on the frame