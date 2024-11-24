import tensorflow as tf
import matplotlib.pyplot as plt

print("\n##################################################################")
# Set memory limit to 8GB for the first visible GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Limit GPU memory usage to 8GB
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)],
        )
        print("GPU memory limit set to 8GB.")
    except RuntimeError as e:
        print(e)
print("##################################################################\n")

from data import process_path
from data import EPOCHS, BATCH_SIZE
from training import generator, discriminator, train


tf.config.run_functions_eagerly(True)

image_paths = (
    tf.data.Dataset.list_files("dataset/ffhq/*.jpg")
    .concatenate(tf.data.Dataset.list_files("dataset/celeba/*.jpg"))
    .concatenate(tf.data.Dataset.list_files("dataset/sfhq_pt1/*.jpg"))
    .concatenate(tf.data.Dataset.list_files("dataset/sfhq_pt2/*.jpg"))
    .concatenate(tf.data.Dataset.list_files("dataset/sfhq_pt3/*.jpg"))
    .concatenate(tf.data.Dataset.list_files("dataset/sfhq_pt4/*.jpg"))
)
# 525258 images

# Load, preprocess, and batch images
train_dataset = image_paths.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = (
    train_dataset.shuffle(buffer_size=20000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Display the generator's and discriminator's architecture.
generator.summary()
discriminator.summary()

# Train the model and save the generator's and discriminator's
# losses over all epochs for visualization.
gen_losses, disc_losses = train(train_dataset, EPOCHS)

# Save the models.
generator.save("generator.keras")
discriminator.save("discriminator.keras")
