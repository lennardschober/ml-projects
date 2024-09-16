import keras
import numpy as np

import helper


# ---------------------------------------------------------------------------------------------------------------------
# -- GLOBAL VARIABLES -------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
input_dir = helper.eval_input_dir               # directory of training images
annotations_file = helper.eval_annotations_file # file containing annotations
sequence_length = helper.sequence_length        # output sequence length
characters = helper.characters                  # character set

# load the saved model
my_model = keras.models.load_model('captcha_solver.keras', safe_mode=False)


# ---------------------------------------------------------------------------------------------------------------------
# -- DATA PREPARATION -------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# Convert images and labels to NumPy arrays with additional (batch) dimension
image_data, label_data= helper.load_images_and_labels(annotations_file)
image_data = np.expand_dims(image_data, axis=-1)
label_data = np.expand_dims(label_data, axis=-1)


# ---------------------------------------------------------------------------------------------------------------------
# -- PREDICT ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def evaluate(images, labels, num_samples=500):
    """
    Evaluates the model on a given set of images and labels. Compares predictions to ground truth and prints
    the number of incorrect predictions and characters.

    Parameters:
    images (list of np.ndarray): List of images to evaluate.
    labels (list of np.ndarray): Corresponding labels (encoded) for each image.

    Returns:
    None
    """

    # Randomly sample indices
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Select the subset of images and labels
    sampled_images = [images[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]
    
    num_images = len(sampled_images)
    num_chars = 5 * len(sampled_labels)
    wrong_pred = 0
    wrong_chars = 0
    
    for img, lbl in zip(sampled_images, sampled_labels):  # Use zip to iterate over images and labels simultaneously
        prediction = my_model.predict(np.expand_dims(img, axis=0))  # Predict for a single image (add batch dimension)
        prediction = helper.decode_prediction(prediction)  # Decode the predicted label
        lbl = helper.decode_label(lbl)
        if prediction != lbl:
            wrong_pred += 1
            wrong_chars += sum(p != l for p, l in zip(prediction, lbl))
            print("--------------------------------------------")
            print("False prediction: ", prediction)
            print("Ground truth:     ", lbl)  # Print prediction and ground truth
    
    print("\n--------------------------------------------")
    print("Solved captchas:   ", num_images - wrong_pred, "/", num_images)
    print("Correct characters: ", num_chars - wrong_chars, "/", num_chars)
    print("Accuracy: ", 100 - 100 * wrong_chars / num_chars, "%.")


evaluate(image_data, label_data)