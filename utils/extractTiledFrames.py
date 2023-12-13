import cv2
import math

# Function to slice the frame into 10x10 tiles
def slice_frame_to_tiles(frame, num_tiles=10):
    tiles = []
    height, width, _ = frame.shape
    M, N = height // num_tiles, width // num_tiles
    for y in range(0, height, M):
        for x in range(0, width, N):
            tile = frame[y:y+M, x:x+N]
            tiles.append(tile)
    return tiles

# Load the video
video_path = '/Users/apaula/Downloads/20230314_110636077_blue_DJI_0149_trimmed.mov'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# FPS and frame interval settings
fps = 50  # Frames per second
frame_interval = 2  # Interval in seconds for frame capture

# Process video
current_time = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Check if it's time to capture the frame
    if math.isclose(current_time % frame_interval, 0, abs_tol=1e-2):
        tiles = slice_frame_to_tiles(frame)
        # Save each tile
        for i, tile in enumerate(tiles):
            filename = f'frame_{math.floor(current_time)}_tile_{i}.jpg'
            cv2.imwrite(filename, tile)

    current_time += 1 / fps

# Release the video capture object
cap.release()
