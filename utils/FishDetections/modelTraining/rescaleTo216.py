import os
from PIL import Image

def format_label(class_id, x_center, y_center, width, height):
    """
    Format the label to match the original format:
    - Class ID as an integer
    - Coordinates and dimensions with the same precision as original
    """
    return f"{int(class_id)} {x_center:.12f} {y_center:.12f} {width:.12f} {height:.12f}"

def updated_crop_and_adjust_annotations(image_path, label_path, output_image_path, output_label_path, crop_left):
    """
    Updated function to crop the image, adjust annotations, and format the labels.
    """
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        if crop_left:
            cropped_img = img.crop((0, 0, 216, img_height))
        else:
            cropped_img = img.crop((img_width - 216, 0, img_width, img_height))
        cropped_img.save(output_image_path)

    with open(label_path, 'r') as file:
        lines = file.readlines()

    adjusted_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id, x_center, y_center, width, height = map(float, parts)
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            if crop_left:
                if x_center + width / 2 > 216:
                    continue
            else:
                if x_center - width / 2 < img_width - 216:
                    continue
                x_center -= (img_width - 216)

            x_center /= 216
            y_center /= img_height
            width /= 216
            height /= img_height

            formatted_label = format_label(class_id, x_center, y_center, width, height)
            adjusted_lines.append(formatted_label)

    with open(output_label_path, 'w') as file:
        file.write('\n'.join(adjusted_lines))

def process_cropping_and_cleanup(dataset_path):
    for folder in ['train', 'valid', 'test']:
        images_path = os.path.join(dataset_path, folder, 'images')
        labels_path = os.path.join(dataset_path, folder, 'labels')

        for image_file in os.listdir(images_path):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(images_path, image_file)
                label_path = os.path.join(labels_path, image_file.replace('.jpg', '.txt'))

                if not os.path.exists(label_path):
                    continue

                # Cropping and adjusting annotations for left and right images
                left_cropped_image_path = os.path.join(images_path, f"left_{image_file}")
                left_cropped_label_path = os.path.join(labels_path, f"left_{image_file.replace('.jpg', '.txt')}")
                updated_crop_and_adjust_annotations(image_path, label_path, left_cropped_image_path, left_cropped_label_path, crop_left=True)

                right_cropped_image_path = os.path.join(images_path, f"right_{image_file}")
                right_cropped_label_path = os.path.join(labels_path, f"right_{image_file.replace('.jpg', '.txt')}")
                updated_crop_and_adjust_annotations(image_path, label_path, right_cropped_image_path, right_cropped_label_path, crop_left=False)

                # Deleting the original image and annotation
                os.remove(image_path)
                os.remove(label_path)

    print("Cropping, annotations adjustment, and cleanup complete.")

# Re-processing the dataset with updated functions
process_cropping_and_cleanup('/Users/apaula/Downloads/fishschool-6')
