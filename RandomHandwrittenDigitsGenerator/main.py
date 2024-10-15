import tensorflow as tf
from keras import datasets
import matplotlib.pyplot as plt

from training import train, generator, discriminator
from data import plot_losses, create_gif
from data import BUFFER_SIZE, BATCH_SIZE, EPOCHS


print("\n##################################################################")
print("Available GPUs:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU available, using CPU.")
print("##################################################################\n")

# Load the MNIST dataset.
(train_images, train_labels), (x_test, y_test) = datasets.mnist.load_data()

# Ensure the images are 28x28x1 (grayscale) and normalize the images to [-1, 1].
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# Batch and shuffle the data.
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Display the generator's and discriminator's architecture.
generator.summary()
discriminator.summary()

# Train the model and save the generator's and discriminator's
# losses over all epochs for visualization.
gen_losses, disc_losses = train(train_dataset, EPOCHS)

# Save the models.
generator.save('generator.keras')
discriminator.save('discriminator.keras')

# Plot the losses.
plot_losses(gen_losses, disc_losses)

# Save the progress gif.
create_gif()