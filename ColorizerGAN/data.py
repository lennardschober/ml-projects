import cv2
import tensorflow as tf

EPOCHS = 3  # Number of epochs
BATCH_SIZE = 25  # Batch size
IMG_SIZE = 256  # Image has size of (IMG_SIZE x IMG_SIZE)
DECAY_START_EPOCH = EPOCHS // 2  # Epoch on which to start with linear decay
OUTPUT_CHANNELS = 2  # Number of channels G should output
MODEL_SAVE_INTERVAL = 5  # Save the models every MODEL_SAVE_INTERVAL epochs


# Define function to load and process images
def load_and_preprocess_image(image_path):
    # Decode the image file path to string if necessary (e.g., on EagerTensors)
    image_path = image_path.numpy().decode("utf-8")
    bgr_image = cv2.imread(
        image_path, cv2.IMREAD_COLOR
    )  # Load in BGR format by default

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


# Wrap function for use in tf.data.Dataset
def process_path(file_path):
    return tf.py_function(load_and_preprocess_image, [file_path], [tf.float32])
