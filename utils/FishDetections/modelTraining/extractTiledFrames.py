import cv2
import math
import os
import numpy as np

# Function to extract tiles of specified size from a frame at random positions
def extract_random_tiles(frame, tile_size, num_tiles):
    tiles = []
    height, width, _ = frame.shape
    
    for _ in range(num_tiles):
        y = np.random.randint(0, height - tile_size + 1)
        x = np.random.randint(0, width - tile_size + 1)
        tile = frame[y:y+tile_size, x:x+tile_size]
        tiles.append(tile)
    
    return tiles

# Load the video
video_path = '/Volumes/T7-August/trex/20240220_175636912_DJI_20240220175636_0003_V.MP4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Extract the video name for folder creation
video_name = video_path.split('/')[-1].split('.')[0]
tiles_folder = f'{video_name}_tiles9'
os.makedirs(tiles_folder, exist_ok=True)

# Extract 'DJI_20240220175636' part from the video name
dji_part = video_name.split('_')[2]

# Settings for tile extraction
tile_size = 320  # Tile size
num_tiles_per_frame = 5  # Number of tiles to extract per selected frame
selected_frames = np.random.choice(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), size=300, replace=False)

# Process video
current_frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Check if the current frame is one of the selected frames
    if current_frame_index in selected_frames:
        tiles = extract_random_tiles(frame, tile_size, num_tiles_per_frame)
        
        # Save each tile
        for i, tile in enumerate(tiles):
            filename = f'{tiles_folder}/{dji_part}_frame_{current_frame_index}_tile_{i}.jpg'
            cv2.imwrite(filename, tile)
    
    current_frame_index += 1

# Release the video capture object
cap.release()
