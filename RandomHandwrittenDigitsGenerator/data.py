import os

import imageio
import matplotlib.pyplot as plt


# =================
# === VARIABLES ===
# =================
BUFFER_SIZE = 60000             # The whole dataset will be considered for shuffling.
BATCH_SIZE = 512
EPOCHS = 400
noise_dim = 200                 # Size of the input noise.
num_examples_to_generate = 16   # Number of images for the gif.


# =================
# === FUNCTIONS ===
# =================
def generate_and_save_images(model, epoch, test_input):
    # 'training' is set to false so the model is run in inference mode.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('img/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def plot_losses(gen_losses, disc_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.title('Generator and Discriminator Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def create_gif(image_folder='img', gif_name='progress', duration=10.0, step=1):
    # Get a sorted list of all image file paths in the folder.
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

    # Load every nth image (step)
    image_list = []
    for i, filename in enumerate(images):
        if i % step == 0:
            img_path = os.path.join(image_folder, filename)
            image_list.append(imageio.imread(img_path))

    # Save the selected images as a GIF.
    gif_path = f'{gif_name}.gif'
    imageio.mimsave(gif_path, image_list, duration=duration)
    print(f"GIF saved at {gif_path}")