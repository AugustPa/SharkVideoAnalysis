import json
import numpy as np

def load_detections(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_sampled_properties(detections, frame_shape, radius, step):
    height, width = frame_shape[:2]
    sampled_properties = []  # To store properties for sampled points

    class_to_angle = {
        0: 0,                 # 'east'
        2: np.pi / 4,         # 'northeast'
        1: np.pi / 2,         # 'north'
        3: 3 * np.pi / 4,     # 'northwest'
        7: np.pi,             # 'west'
        6: 5 * np.pi / 4,     # 'southwest'
        4: 3 * np.pi / 2,     # 'south'
        5: 7 * np.pi / 4      # 'southeast'
    }

    def in_circle(center_x, center_y, radius, x, y):
        return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

    for y in range(0, height, step):
        for x in range(0, width, step):
            region_orientations = []
            for box, cls in zip(detections['boxes'], detections['classes']):
                box_center_x, box_center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if in_circle(x, y, radius, box_center_x, box_center_y):
                    region_orientations.append(class_to_angle[cls])

            if region_orientations:
                mean_orientation = np.mean(region_orientations)
                polarization = np.std(region_orientations)
                density = len(region_orientations) / (np.pi * radius ** 2)
                sampled_properties.append({'x': x, 'y': y, 'density': density, 
                                           'mean_orientation': mean_orientation, 
                                           'polarization': polarization})

    return sampled_properties

# Example usage
json_path = '/Users/apaula/src/SharkVideoAnalysis/detections.json'  # Path to your JSON file
frame_shape = (2160,3840,3)  # Example frame shape, adjust to your actual frame size
radius = 500  # Example radius
step = 5    # Sample every 10th pixel

detections_data = load_detections(json_path)
for frame_index, detections in enumerate(detections_data):
    properties = calculate_sampled_properties(detections, frame_shape, radius, step)
    # Save properties to a file
    with open(f'frame_{frame_index}_properties.json', 'w') as f:
        json.dump(properties, f)

    print(f"Properties for frame {frame_index} saved.")