import os
import numpy as np
from PIL import Image

def rotate_point(x, y, angle, cx, cy):
    x_rotated = cx + np.cos(angle) * (x - cx) - np.sin(angle) * (y - cy)
    y_rotated = cy + np.sin(angle) * (x - cx) + np.cos(angle) * (y - cy)
    return x_rotated, y_rotated

def update_class_id(class_id, angle_degrees):
    # 90-degree rotation mappings
    label_mapping_90 = {0: 4, 1: 0, 2: 5, 3: 2, 4: 7, 5: 6, 6: 3, 7: 1}
    # 180-degree rotation mappings
    label_mapping_180 = {0: 7, 1: 4, 2: 6, 3: 5, 4: 1, 5: 3, 6: 2, 7: 0}
    # 270-degree rotation mappings
    label_mapping_270 = {0: 1, 1: 7, 2: 3, 3: 6, 4: 0, 5: 2, 6: 5, 7: 4}

    if angle_degrees == 90:
        return label_mapping_90.get(class_id, class_id)
    elif angle_degrees == 180:
        return label_mapping_180.get(class_id, class_id)
    elif angle_degrees == 270:
        return label_mapping_270.get(class_id, class_id)
    else:
        return class_id
def get_rotated_bbox(x_center, y_center, width, height, angle, img_width, img_height):
    # Convert center and size to corner coordinates
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # List of corners
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    # Rotate each corner
    rotated_corners = [rotate_point(x, y, angle, img_width / 2, img_height / 2) for x, y in corners]

    # Find min and max points
    min_x = min([corner[0] for corner in rotated_corners])
    max_x = max([corner[0] for corner in rotated_corners])
    min_y = min([corner[1] for corner in rotated_corners])
    max_y = max([corner[1] for corner in rotated_corners])

    # Convert back to center and size format
    new_width = max_x - min_x
    new_height = max_y - min_y
    new_x_center = min_x + new_width / 2
    new_y_center = min_y + new_height / 2

    return new_x_center, new_y_center, new_width, new_height

def process_annotation_line(line, angle_radians, angle_degrees, img_width, img_height):
    parts = line.split()
    if len(parts) != 5:
        return line

    class_id, x_center, y_center, width, height = map(float, parts)
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    new_x_center, new_y_center, new_width, new_height = get_rotated_bbox(
        x_center, y_center, width, height, angle_radians, img_width, img_height)

    class_id = update_class_id(int(class_id), angle_degrees)
    new_x_center /= img_width
    new_y_center /= img_height
    new_width /= img_width
    new_height /= img_height

    return f"{class_id} {new_x_center} {new_y_center} {new_width} {new_height}"

def process_dataset(dataset_path):
    for folder in ['train', 'valid', 'test']:
        images_path = os.path.join(dataset_path, folder, 'images')
        labels_path = os.path.join(dataset_path, folder, 'labels')

        for image_file in os.listdir(images_path):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(images_path, image_file)
                label_path = os.path.join(labels_path, image_file.replace('.jpg', '.txt'))

                if not os.path.exists(label_path):
                    continue

                with open(label_path, 'r') as file:
                    label_contents = file.read()

                image = Image.open(image_path)

                for angle_degrees in [90, 180, 270]:
                    angle_radians = np.radians(angle_degrees)
                    rotated_image = image.rotate(-angle_degrees, expand=True)

                    rotated_annotations = []
                    for line in label_contents.split('\n'):
                        if line.strip():
                            updated_line = process_annotation_line(line, angle_radians, angle_degrees, image.width, image.height)
                            rotated_annotations.append(updated_line)

                    # Save rotated image and annotations
                    rotated_image_file = f"{os.path.splitext(image_file)[0]}_rotated_{angle_degrees}.jpg"
                    rotated_label_file = f"{os.path.splitext(image_file)[0]}_rotated_{angle_degrees}.txt"

                    rotated_image_path = os.path.join(images_path, rotated_image_file)
                    rotated_label_path = os.path.join(labels_path, rotated_label_file)

                    rotated_image.save(rotated_image_path)
                    with open(rotated_label_path, 'w') as file:
                        file.write('\n'.join(rotated_annotations))

# Path to your dataset directory
dataset_path = '/Users/apaula/Downloads/fishschool-5'
process_dataset(dataset_path)

print("Dataset processing complete.")