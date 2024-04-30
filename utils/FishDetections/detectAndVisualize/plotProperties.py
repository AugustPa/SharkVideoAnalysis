import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def load_properties(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def plot_properties(frame_index, frame_shape):
    properties_path = f'frame_{frame_index}_properties.json'
    properties = load_properties(properties_path)

    # Create empty arrays for density, polarization, and mean orientation
    density_map = np.zeros(frame_shape[:2])
    polarization_map = np.zeros(frame_shape[:2])
    orientation_map = np.zeros(frame_shape[:2])

    # Fill in the maps
    for prop in properties:
        x, y = int(prop['x']/10), int(prop['y']/10)
        density = prop['density']
        polarization = prop['polarization']
        orientation = prop['mean_orientation']

        x = min(x, frame_shape[1] - 1)
        y = min(y, frame_shape[0] - 1)

        density_map[y, x] = density
        polarization_map[y, x] = polarization
        orientation_map[y, x] = orientation

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Density Heatmap
    axes[0].imshow(density_map, cmap='viridis', extent=[0, frame_shape[1], frame_shape[0], 0])
    axes[0].set_title('Density')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # Polarization Heatmap
    axes[1].imshow(polarization_map, cmap='plasma', extent=[0, frame_shape[1], frame_shape[0], 0])
    axes[1].set_title('Polarization')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')

    # Mean Orientation Heatmap
    im = axes[2].imshow(orientation_map, cmap='hsv', extent=[0, frame_shape[1], frame_shape[0], 0])
    axes[2].set_title('Mean Orientation')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')

    # Add a colorbar for the Mean Orientation
    fig.colorbar(im, ax=axes[2], orientation='vertical', label='Mean Orientation (Radians)')

    plt.tight_layout()
    plt.show()

def plot_properties_2(frame_index, frame_shape):
    properties_path = f'frame_{frame_index}_properties.json'
    properties = load_properties(properties_path)

    # Initialize an empty HSV image
    hsv_image = np.zeros((frame_shape[0], frame_shape[1], 3))
    
    # Normalize density (if necessary)
    max_density = max([prop['density'] for prop in properties]) if properties else 1


    for prop in properties:
        x, y = int(prop['x']/10), int(prop['y']/10)
        density = prop['density'] / max_density  # Normalizing density
        polarization = prop['polarization']
        orientation = prop['mean_orientation']

        x = min(x, frame_shape[1] - 1)
        y = min(y, frame_shape[0] - 1)

        # Map properties to HSV
        hue = orientation * 180 / np.pi  # Convert to degrees
        saturation = polarization
        value = density

        hsv_image[y, x, 0] = hue
        hsv_image[y, x, 1] = saturation
        hsv_image[y, x, 2] = value

    # Convert HSV to RGB for plotting
    rgb_image = cv2.cvtColor(hsv_image.astype('float32'), cv2.COLOR_HSV2RGB)

    # Plot
    plt.imshow(rgb_image)
    plt.title('Combined Properties Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Example usage
frame_shape = (215, 307, 3)  # Adjust to your frame shape
plot_properties(frame_index=0, frame_shape=frame_shape)
