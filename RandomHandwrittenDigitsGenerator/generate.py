import random
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from data import noise_dim


# Load the saved generator model.
generator = tf.keras.models.load_model('generator.keras')

def generate_grid(dim=4):
    # Generate random noise vector for a batch of dim^2 images.
    random_noise = np.random.normal(0, 1, (dim * dim, noise_dim))

    # Generate images.
    generated_images = generator.predict(random_noise)

    # Scale pixel values to [0, 1].
    generated_images = 0.5 * generated_images + 0.5  

    # Create a dim*dim grid of images using matplotlib.
    fig, axes = plt.subplots(dim, dim, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):
        img = generated_images[i, :, :, 0]  # Remove the batch dimension and extract the 2D image.
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # Hide axes.

    plt.tight_layout()
    plt.show()

    # Optional: Save the grid of images as a single image (concatenate them).
    # Create a blank canvas for the dim*dim grid.
    grid_image = np.zeros((28 * dim, 28 * dim)) # Assuming the images are 28x28 pixels.

    for i in range(dim):
        for j in range(dim):
            grid_image[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = generated_images[i * dim + j, :, :, 0]

    # Convert the grid image to uint8 and save using PIL.
    grid_image_pil = Image.fromarray((grid_image * 255).astype(np.uint8), mode='L')  # 'L' mode for grayscale.
    grid_image_pil.save('generated_image_grid.png')


def find_bounding_box(image):
    """Find the bounding box of the non-zero pixels (the digit) in the image."""
    positions = np.nonzero(image)

    ymin = 0    # We are only interested
    ymax = 27   # in the x coordinates.
    xmin = positions[1].min()
    xmax = positions[1].max()

    return xmin, xmax, ymin, ymax


def generate_number(num_digits=4, min_spacing=1, max_spacing=5, thousands_spacing=10):
    # Generate random noise vector for a batch of num_digits images.
    random_noise = np.random.normal(0, 1, (num_digits, noise_dim))  # num_digits = number of images.

    # Generate images from the noise.
    generated_images = generator.predict(random_noise)

    # Scale pixel values from [-1, 1] to [0, 1].
    generated_images = 0.5 * generated_images + 0.5  

    # Create an empty list to store cropped digits.
    cropped_digits = []

    for i in range(num_digits):
        # Extract the individual image from the batch.
        # Cast to [0, 255] to calculate bounding box.
        image = (generated_images[i, :, :, 0] * 255).astype(np.uint8)
        # Find the bounding box of the digit.
        xmin, xmax, ymin, ymax = find_bounding_box(image)
        
        # Crop the image to the bounding box.
        cropped_digit = image[ymin:ymax+1, xmin:xmax+1]
        cropped_digits.append(cropped_digit)


    # Compute the random spacing and save for later.
    spacing = []
    for i in range(num_digits - 1):
        if (i + 1) % 3 == 0:
            spacing.insert(0, random.randint(min_spacing, max_spacing) + thousands_spacing)
        else:
            spacing.insert(0, random.randint(min_spacing, max_spacing))

    # Determine the total width of the canvas by summing the widths of the cropped digits and random spacing.
    total_width = sum(digit.shape[1] for digit in cropped_digits) + sum(spacing)

    # Create a blank canvas with a height of the largest digit and total calculated width.
    max_height = max(digit.shape[0] for digit in cropped_digits)
    grid_image = np.zeros((max_height, total_width))  # Empty canvas.

    current_x = 0  # Starting x position for placing the first digit.

    for i, cropped_digit in enumerate(cropped_digits):
        # Get the height and width of the current cropped digit.
        h, w = cropped_digit.shape

        # Place the cropped digit on the canvas at the current x position, vertically centered.
        grid_image[(max_height - h) // 2:(max_height - h) // 2 + h, current_x:current_x + w] = cropped_digit

        # Random spacing for the next digit.
        if i < num_digits - 1:
            current_x += w + spacing[i]

    # Convert the grid image to uint8 and save using PIL.
    grid_image_pil = Image.fromarray((grid_image).astype(np.uint8), mode='L')  # 'L' mode for grayscale.
    if num_digits == 1:
        grid_image_pil.save(f'generated_number_with_{num_digits}_digit.png')
    else:
        grid_image_pil.save(f'generated_number_with_{num_digits}_digits.png')


generate_grid()
generate_number()