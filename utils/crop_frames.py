import cv2
import numpy as np
import os

# Conversion rate from cm to pixel and frame dimensions
conversion_rate = 0.129291
frame_height = 2160  # The height of the frames in the video
frame_width = 3840  # The width of the frames in the video

# Debug: Print OpenCV version
print("OpenCV Version:", cv2.__version__)

# Load the video file using default backend
video = cv2.VideoCapture('20230320_122211654_red_DJI_0310.MP4')

# Debug: Check if video file is opened successfully
if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

# Load the centroid coordinates
try:
    npz_file_name = '20230320_122211654_red_DJI_0310_fish12.npz'
    npz_data = np.load(npz_file_name)
except Exception as e:
    print("Error: Could not load .npz file. ", e)
    exit()

x_centroid = npz_data['X#wcentroid'] / conversion_rate
y_centroid = npz_data['Y#wcentroid'] / conversion_rate

# Create output directory if it doesn't exist
output_dir = f'cropped_frames_{os.path.splitext(npz_file_name)[0]}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Debug: Initialize frame count
frame_count = 0

# JPEG compression parameters
jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 90]

# Loop through each frame and centroid coordinate to crop the frame
for i, (x, y) in enumerate(zip(x_centroid, y_centroid)):
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Skip frames where x or y is invalid (NaN, inf, -inf)
    if not np.isfinite(x) or not np.isfinite(y):
        continue

    # Debug: Increment frame count
    frame_count += 1

    # Calculate crop window top left corner (making sure it's within frame dimensions)
    x1 = int(x - 800)
    y1 = int(y - 800)
    
    # Calculate padding if cropping window goes out of frame dimensions
    pad_left = max(0, -x1)
    pad_right = max(0, x1 + 1600 - frame_width)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y1 + 1600 - frame_height)
    
    # Apply padding
    x1 += pad_left
    y1 += pad_top
    
    # Crop and pad the frame
    cropped_frame = frame[y1:y1+1600-pad_top-pad_bottom, x1:x1+1600-pad_left-pad_right]
    padded_frame = cv2.copyMakeBorder(cropped_frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    # Save the cropped frame as compressed JPEG
    output_path = os.path.join(output_dir, f'frame_{i}.jpg')
    cv2.imwrite(output_path, padded_frame, jpeg_params)

# Release the video capture object
video.release()

# Debug: Print total frame count
print(f"Total frames processed: {frame_count}")
