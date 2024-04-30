import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def test_tiling_on_frame(video_path, frame_number=100, tile_size=640, overlap=0.2):
    # Load a specific frame from the video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to load the frame from the video.")
        return

    # Apply tiling
    tiles = tile_image(frame, tile_size, overlap)

    # Plot the frame with tiles overlaid
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for _, position in tiles:
        x, y = position
        rect = patches.Rectangle((x, y), tile_size, tile_size, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.axis('off')
    plt.show()

# Example usage
video_path = '/Users/apaula/Downloads/fishschool_trimmed.mp4'  # Replace with your video path
test_tiling_on_frame(video_path)
