import time

import cv2
import tensorflow as tf
from IPython import display
from keras import optimizers, losses

from data import EPOCHS, DECAY_START_EPOCH, MODEL_SAVE_INTERVAL
from model import build_unet_generator, build_patchgan_discriminator


# Define models.
generator = build_unet_generator()
discriminator = build_patchgan_discriminator()


# Loss function for the generator
def generator_loss(fake_output, generated_image, target_image, LAMBDA=100):
    # Adversarial loss
    adversarial_loss = losses.binary_crossentropy(
        tf.ones_like(fake_output), fake_output, from_logits=True
    )

    # Extract a and b channels (channels 1 and 2)
    generated_ab = generated_image[..., 1:]  # Shape: [batch_size, height, width, 2]
    target_ab = target_image[..., 1:]  # Shape: [batch_size, height, width, 2]

    # L1 loss between a and b channels of generated and target images
    l1_loss_ab = tf.reduce_mean(tf.abs(generated_ab - target_ab))

    # Total generator loss
    return LAMBDA * l1_loss_ab + adversarial_loss


# Discriminator loss function
def discriminator_loss(real_output, generated_output):
    # Loss for real samples
    real_loss = losses.binary_crossentropy(
        tf.ones_like(real_output), real_output, from_logits=True
    )

    # Loss for fake samples
    generated_loss = losses.binary_crossentropy(
        tf.zeros_like(generated_output), generated_output, from_logits=True
    )

    # Total discriminator loss
    return (real_loss + generated_loss) / 2


# Define both models optimizers.
global_lr = 2e-4
generator_optimizer = optimizers.Adam(learning_rate=global_lr, beta_1=0.5)
discriminator_optimizer = optimizers.Adam(learning_rate=global_lr, beta_1=0.5)


# Define a custom learning rate schedule
def update_LR(epoch):
    global global_lr

    if epoch < DECAY_START_EPOCH:
        return

    reduction = global_lr / (EPOCHS - DECAY_START_EPOCH + 1)

    gen_new_lr = generator_optimizer.learning_rate - reduction
    generator_optimizer.learning_rate.assign(gen_new_lr)
    disc_new_lr = discriminator_optimizer.learning_rate - reduction
    discriminator_optimizer.learning_rate.assign(disc_new_lr)

    print(
        f"\n # Learning rate of generator set to     {gen_new_lr:.7g} for epoch {epoch + 1}"
    )
    print(
        f"\n # Learning rate of discriminator set to {disc_new_lr:.7g} for epoch {epoch + 1}"
    )


def augment_image_batch(images):
    # Randomly flip each image in the batch horizontally
    images = tf.map_fn(tf.image.random_flip_left_right, images)

    # Randomly rotate each image by 0, 90, or 270 degrees
    # random_rotation = np.random.choice([0, 0, 1, 3])  # 0: 0°, 1: 90°, 3: 270°
    # Apply random rotation to each image in the batch
    # images = tf.image.rot90(images, k=random_rotation)

    return images


# '@tf.function' causes the function to be "compiled".
@tf.function
def train_step(img_l, real_lab):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Augment for G
        generated_l = augment_image_batch(img_l)
        # Generate fake ab channels
        generated_ab = generator(generated_l, training=True)
        generated_lab = tf.concat([generated_l, generated_ab], axis=-1)

        # Augment and split real images for D
        real_lab = augment_image_batch(real_lab)
        real_l = real_lab[..., :1]  # Extract ab channels from real image
        real_ab = real_lab[..., 1:]  # Extract ab channels from real image

        fake_output = discriminator([generated_l, generated_ab], training=True)
        real_output = discriminator([real_l, real_ab], training=True)

        G_loss = generator_loss(fake_output, generated_lab, real_lab)
        D_loss = discriminator_loss(real_output, fake_output)

    generator_gradients = gen_tape.gradient(G_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(
        D_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    return tf.reduce_mean(G_loss).numpy(), tf.reduce_mean(D_loss).numpy()


def load_predefined_image(image_path):
    # Load grayscale image
    sample_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Scale to [-1, 1] and return as a tensor
    sample_image = tf.convert_to_tensor(sample_image, dtype=tf.float32)
    sample_image = (sample_image / 127.5) - 1
    return sample_image


# Function to save generated images using a sample from the dataset
def save_generated_image(
    generator, sample_image, epoch, counter, folder_path="PROGRESS"
):
    # Prepare the sample image for the generator
    sample_image2 = tf.reshape(sample_image, (1, 256, 256, 1))

    # Generate an image using the generator
    generated_ab = generator(sample_image2, training=False)
    generated_lab = tf.concat(
        [sample_image2, generated_ab], axis=-1
    )  # shape (1, 256, 256, 3)
    generated_lab = tf.reshape(generated_lab, (256, 256, 3))

    # Ensure the folder exists
    tf.io.gfile.makedirs(folder_path)

    # Scale LAB image back to [0, 255] for OpenCV conversion
    generated_lab = (generated_lab + 1) * 127.5
    generated_lab = tf.cast(generated_lab, tf.uint8).numpy()

    # Convert from LAB to RGB using OpenCV
    rgb_image = cv2.cvtColor(generated_lab, cv2.COLOR_LAB2RGB)

    # Encode as PNG
    encoded_image = tf.image.encode_png(rgb_image)

    # Write image file
    tf.io.write_file(
        f"{folder_path}/generated_ep_{epoch + 1}_img_{counter}.png", encoded_image
    )


# define the image on which to track progress
predefined_image_path = "YOURPATH"
sample_image = load_predefined_image(predefined_image_path)


# Training loop.
def train(dataset, epochs):
    # Initialize lists to keep track of the losses.
    gen_loss_list = []
    disc_loss_list = []
    gen_loss_list_CALLBACK = []
    disc_loss_list_CALLBACK = []

    # image_save_interval = 875  # Set the interval to save images every 5,000 images
    image_save_interval = 2101  # Set the interval to save images every 10,000 images
    image_counter = 0  # Counter to track the number of images processed
    saved_counter = 0

    for epoch in range(epochs):
        start = time.time()

        # Reduce learning rate
        update_LR(epoch)

        # Initialize variables to accumulate losses for this epoch.
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        num_batches = 0

        for img_lab in dataset:
            # Get grayscale from img_lab
            img_lab = tf.squeeze(img_lab, axis=0)
            img_l, _, _ = tf.split(img_lab, num_or_size_splits=3, axis=-1)

            # Perform training step and capture the losses.
            gen_loss, disc_loss = train_step(img_l, img_lab)
            total_gen_loss += gen_loss
            total_disc_loss += disc_loss
            num_batches += 1
            image_counter += 1  # Increment the counter

            # Save the generated image every few processed images
            if image_counter >= image_save_interval:
                saved_counter += 1
                save_generated_image(generator, sample_image, epoch, saved_counter)
                image_counter = 0  # Reset the counter after saving

        # Average the losses over the number of batches.
        avg_gen_loss = total_gen_loss / num_batches
        avg_disc_loss = total_disc_loss / num_batches

        # Add losses to their lists for tracking progress.
        gen_loss_list.append(avg_gen_loss)
        disc_loss_list.append(avg_disc_loss)

        # Optionally save the models.
        if epoch != 0 and epoch % MODEL_SAVE_INTERVAL == 0:
            generator.save(f"YOURPATH/generator_{epoch}.keras")
            discriminator.save(f"YOURPATH/discriminator_{epoch}.keras")

        # Print info of the latest epoch in form of time taken and both losses.
        display.clear_output(wait=True)
        print("\n-----------------------------")
        print("Time for epoch {:4d} is {:5.2f}s".format(epoch + 1, time.time() - start))
        print("     Generator Loss:      {:.5f}".format(avg_gen_loss))
        print("     Discriminator Loss:  {:.5f}".format(avg_disc_loss))

    # Generate after the final epoch.
    display.clear_output(wait=True)

    return gen_loss_list, disc_loss_list
