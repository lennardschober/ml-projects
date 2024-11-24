import os
import random

import cv2
import numpy as np
import tensorflow as tf
from keras import models
import matplotlib.pyplot as plt


# Load the trained generator model
generator = models.load_model("generator.keras")


def preprocess_image(image_path):
    # Decode the image file path to string if necessary (e.g., on EagerTensors)

    bgr_image = cv2.imread(
        image_path, cv2.IMREAD_COLOR
    )  # Load in BGR format by default
    bgr_image = cv2.resize(bgr_image, (256, 256))
    # Convert BGR to LAB color space
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

    # Convert L to range [-1, 1]
    lab_image = lab_image.astype("float32")
    lab_image[..., 0] = (
        lab_image[..., 0] / 127.5
    ) - 1  # Scale L from [0, 255] to [-1, 1]
    lab_image[..., 1:] = (
        lab_image[..., 1:] - 128
    ) / 127.5  # Scale A and B from [0, 255] to [-1, 1]

    # Convert to tensor
    lab_image = tf.convert_to_tensor(lab_image, dtype=tf.float32)

    return lab_image


def generate_images_with_labels(input_folder, output_path, generator):
    """Generate and save a grid with black-and-white, generated, and ground truth images, including row labels."""
    # Collect all image paths in the folder
    image_paths = [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    random.seed(1)
    random.shuffle(image_paths)  # Shuffle the list of paths

    # Store results for plotting
    bw_images = []
    generated_images = []
    ground_truth_images = []

    for image_path in image_paths:
        # Load and preprocess the ground truth image
        lab_image = preprocess_image(image_path)
        img_l, _, _ = tf.split(lab_image, num_or_size_splits=3, axis=-1)
        img_l = tf.expand_dims(img_l, axis=0)  # Add batch dimension

        # Generate colorized image
        generated_image = generator(img_l, training=False)[0].numpy()
        generated_image = tf.concat(
            [img_l[0].numpy(), generated_image], axis=-1
        ).numpy()

        # Prepare images for visualization
        bw_image = ((img_l[0].numpy().squeeze() + 1) * 127.5).astype(np.uint8)
        gen_image = cv2.cvtColor(
            ((generated_image + 1) * 127.5).astype(np.uint8), cv2.COLOR_LAB2RGB
        )
        gt_image = cv2.cvtColor(
            ((lab_image.numpy().squeeze() + 1) * 127.5).astype(np.uint8),
            cv2.COLOR_LAB2RGB,
        )

        bw_images.append(bw_image)
        generated_images.append(gen_image)
        ground_truth_images.append(gt_image)

    # Create the grid of images
    num_cols = len(image_paths) + 1  # +1 for labels column
    fig, axes = plt.subplots(
        3,
        num_cols,
        figsize=((len(image_paths) + 1) * 256 / 100, 3 * 256 / 100),
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    # Add row labels
    row_labels = [
        "Grayscale",
        "Generated",
        "Ground truth",
    ]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(
            0.5, 0.5, label, ha="center", va="center", fontsize=12, fontweight="bold"
        )
        axes[row_idx, 0].axis("off")

    # Fill in the images
    for col_idx, (bw, gen, gt) in enumerate(
        zip(
            bw_images,
            generated_images,
            ground_truth_images,
        ),
        start=1,
    ):
        # Top row: Black-and-white images
        axes[0, col_idx].imshow(bw, cmap="gray")
        axes[0, col_idx].axis("off")

        # Middle row: Concatenated (BW + Generated)
        axes[1, col_idx].imshow(gen)
        axes[1, col_idx].axis("off")

        # Bottom row: Ground truth
        axes[2, col_idx].imshow(gt)
        axes[2, col_idx].axis("off")

    # Remove overall padding/margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save the figure
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=500)
    plt.close()


# Example usage:
input_folder = "DEL/"  # Path to folder with ground truth images
output_path = "generated.jpg"  # Path to save the result grid
generate_images_with_labels(input_folder, output_path, generator)
