import os

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------------------------------------------------
# -- FUNCTIONS --------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def load_images_and_labels(dir_margot, dir_jaime):
    images = []
    labels = []

    def process_images(directory, label):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                try:
                    with Image.open(file_path) as img:
                        img = img.convert('L')  # Ensure the image is in grayscale
                        img = img.resize((218, 355))  # Resize image to 218x355
                        img_array = np.array(img) / 255.0  # Convert to numpy array and normalize
                        images.append(img_array)
                        labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    process_images(dir_margot, 0)
    process_images(dir_jaime, 1)

    return np.array(images), np.array(labels)  # Convert lists to numpy arrays
