import json
import numpy as np
import matplotlib.pyplot as plt

def load_properties(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


def plot_density(frame_index, frame_shape):
    properties_path = f'frame_{frame_index}_properties.json'
    properties = load_properties(properties_path)

    # Create an empty array for density values
    density_map = np.zeros(frame_shape[:2])

    for prop in properties:
        x, y, density = int(prop['x']/10), int(prop['y']/10), prop['density']
        # Clip x and y to be within the frame boundaries
        x = min(x, frame_shape[1] - 1)
        y = min(y, frame_shape[0] - 1)
        density_map[y, x] = density  # Assign density value

    # Plotting the density heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(density_map, cmap='viridis', extent=[0, frame_shape[1], frame_shape[0], 0])
    plt.colorbar(label='Density')
    plt.title(f'Frame {frame_index} - Density Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Example usage
frame_shape = (215, 307, 3)  # Adjust to your frame shape
plot_density(frame_index=0, frame_shape=frame_shape)
