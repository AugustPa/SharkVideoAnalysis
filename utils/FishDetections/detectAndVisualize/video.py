import cv2
import os
import glob

frame_shape = (1200, 3600, 3)  # Update the frame shape to match your images

def create_video_from_images(frame_shape):
    image_folder = 'frame_images'
    output_video = 'output_video.mp4'
    frame_rate = 10  # You can adjust this as needed

    images = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    if not images:
        print("No images found in the directory.")
        return

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_shape[1], frame_shape[0]))

    for image_file in images:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Error reading image: {image_file}")
            continue
        out.write(frame)

    out.release()
    print("Video creation completed.")

# Example usage
create_video_from_images(frame_shape)
