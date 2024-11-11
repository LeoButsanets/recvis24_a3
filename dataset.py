import os
import shutil
import random


def create_small_dataset(src_dataset_path, dest_dataset_path, num_images_per_class=5):
    """
    Create a smaller version of the dataset for quick testing.

    Args:
        src_dataset_path (str): Path to the original dataset containing train_images, val_images, and test_images.
        dest_dataset_path (str): Path where the smaller version of the dataset should be created.
        num_images_per_class (int): Number of images to copy per class.
    """
    # Define the dataset splits
    splits = ["train_images", "val_images", "test_images"]

    # Iterate over each split (train, val, test)
    for split in splits:
        src_split_path = os.path.join(src_dataset_path, split)
        dest_split_path = os.path.join(dest_dataset_path, split)

        # Create the split directory in the destination path
        if not os.path.exists(dest_split_path):
            os.makedirs(dest_split_path)

        # Iterate over each class directory in the split
        for class_name in os.listdir(src_split_path):
            src_class_path = os.path.join(src_split_path, class_name)
            dest_class_path = os.path.join(dest_split_path, class_name)

            # Create the class directory in the destination path
            if not os.path.exists(dest_class_path):
                os.makedirs(dest_class_path)

            # Get a list of all images in the class directory
            images = [img for img in os.listdir(src_class_path) if os.path.isfile(os.path.join(src_class_path, img))]

            # Randomly select a subset of images to copy
            selected_images = random.sample(images, min(num_images_per_class, len(images)))

            # Copy the selected images to the destination directory
            for img in selected_images:
                shutil.copy(os.path.join(src_class_path, img), os.path.join(dest_class_path, img))


# Example usage
src_dataset_path = "C:/Users/leobu/Desktop/data/sketch_recvis2024"
dest_dataset_path = "data/sketch_recvis2024_small"
create_small_dataset(src_dataset_path, dest_dataset_path, num_images_per_class=1)
