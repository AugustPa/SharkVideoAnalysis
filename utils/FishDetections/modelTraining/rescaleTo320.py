
import os
from PIL import Image
def format_label(class_id, x_center, y_center, width, height):
    """
    Format the label to match the original format:
    - Class ID as an integer
    - Coordinates and dimensions with the same precision as original
    """
    return f"{int(class_id)} {x_center:.12f} {y_center:.12f} {width:.12f} {height:.12f}"

def pad_image_to_640(image_path, output_image_path):
    """
    Pad the image to 640x640 by adding empty pixels.
    """
    with Image.open(image_path) as img:
        old_size = img.size
        new_size = (640, 640)
        new_img = Image.new("RGB", new_size)  # Creates a new image with black background
        new_img.paste(img, ((new_size[0] - old_size[0]) // 2,
                            (new_size[1] - old_size[1]) // 2))
        new_img.save(output_image_path)

def adjust_annotations_for_padding(label_path, output_label_path, old_width, old_height):
    """
    Adjust the annotations for the padded image.
    """
    with open(label_path, 'r') as file:
        lines = file.readlines()

    adjusted_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id, x_center, y_center, width, height = map(float, parts)

            # Adjusting coordinates for the padded image
            x_center = (x_center * old_width + (640 - old_width) / 2) / 640
            y_center = (y_center * old_height + (640 - old_height) / 2) / 640
            width *= old_width / 640
            height *= old_height / 640

            formatted_label = format_label(class_id, x_center, y_center, width, height)
            adjusted_lines.append(formatted_label)

    with open(output_label_path, 'w') as file:
        file.write('\n'.join(adjusted_lines))

def process_padding_resizing_and_cleanup(dataset_path):
    for folder in ['train', 'valid', 'test']:
        images_path = os.path.join(dataset_path, folder, 'images')
        labels_path = os.path.join(dataset_path, folder, 'labels')

        for image_file in os.listdir(images_path):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(images_path, image_file)
                label_path = os.path.join(labels_path, image_file.replace('.jpg', '.txt'))

                # Padding and resizing image
                padded_image_path = os.path.join(images_path, f"padded_{image_file}")
                pad_image_to_640(image_path, padded_image_path)

                # Adjusting annotations for the padded image
                padded_label_path = os.path.join(labels_path, f"padded_{image_file.replace('.jpg', '.txt')}")
                adjust_annotations_for_padding(label_path, padded_label_path, 384, 216)  # Original dimensions

                # Deleting the original image and annotation
                os.remove(image_path)
                os.remove(label_path)

    print("Padding, resizing, annotations adjustment, and cleanup complete.")


# Re-processing the dataset with padding and resizing
process_padding_resizing_and_cleanup('/Users/apaula/Downloads/fishschool-6')
