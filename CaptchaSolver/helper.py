import os
import json

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------------------------------------------------
# -- GLOBAL VARIABLES -------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
input_dir = "dataset/"                              # directory of training images
eval_input_dir = "dataset_eval/"                    # alternative directory used for evaluation
annotations_file = "annotations.json"               # json file containing annotations
eval_annotations_file = "annotations_eval.json"     # alternative json file used for evaluation
sequence_length = 5                                 # output sequence length
characters = '123456789abcdefghijklmnopqrstuvwxyz'  # character set

# ---------------------------------------------------------------------------------------------------------------------
# -- HELPER FUNCTIONS -------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def encode_label(label, num_positions=sequence_length, num_classes=len(characters)):
    """
    Transforms a label of the form [a, b, c, d, e] into a vector of size 5*35.
    
    Parameters:
    label (list): A list of characters representing the captcha label.
    num_positions (int): The length of the sequence (default is 5).
    num_classes (int): Number of possible characters (default is 35).

    Returns:
    np.ndarray: A one-hot encoded vector representing the label.
    """

    # validate correctness of input
    if isinstance(label, list) and len(label) == 5:
        # check if each element is a string and is contained in the allowed characters
        for char in label:
            if char not in characters:
                raise ValueError("Input must only contain elements from 'characters'.")
    else:
        raise ValueError("Input must be a list of exactly 5 elements.")

    # maps a given character to its index in 'characters'
    char_to_index = {char: idx for idx, char in enumerate(characters)}

    # initialize an empty vector for the one-hot encoding (length 5*35)
    one_hot_vector = np.zeros(num_positions * num_classes)
    
    # encode each character as a one-hot vector:
    # every position has 0 except the one corresponding to the characters index
    for pos, char in enumerate(label):
        if char in char_to_index:
            index = char_to_index[char]
            one_hot_vector[pos * num_classes + index] = 1
    
    return one_hot_vector


def decode_label(one_hot_vector, num_positions=sequence_length, num_classes=len(characters)):
    """
    Transforms a vector of size 5*35 into a label of the form [a, b, c, d, e].
    
    Parameters:
    one_hot_vector (np.ndarray): The one-hot encoded vector to decode.
    num_positions (int): The length of the sequence (default is 5).
    num_classes (int): Number of possible characters (default is 35).

    Returns:
    list: A list of characters representing the decoded label.
    """

    decoded_label = []
    
    # iterate through each position (5 positions in total)
    for pos in range(num_positions):
        # extract the slice corresponding to the current position
        slice_start = pos * num_classes
        slice_end = slice_start + num_classes
        position_vector = one_hot_vector[slice_start:slice_end]
        
        # find the index of the maximum value in the one-hot encoded vector
        index = np.argmax(position_vector)
        
        # convert the index back to the corresponding character
        decoded_char = characters[index]
        decoded_label.append(decoded_char)
    
    return decoded_label


def decode_prediction(prediction, num_segments=sequence_length, segment_size=len(characters)):
    """
    Transforms the model's raw prediction output into a list of characters (e.g., [a, b, c, d, e]).

    This function assumes the model predicts a flattened vector where each segment corresponds to one character.
    For each segment, the function selects the character with the highest predicted probability.

    Parameters:
    prediction (np.ndarray): The raw prediction output from the model, typically a vector of shape (1, num_segments * segment_size).
    num_segments (int): The number of segments, each representing one character in the sequence (default is the sequence length).
    segment_size (int): The number of possible characters in each segment (default is the number of characters in the character set).

    Returns:
    list: A list of the top predicted characters corresponding to the model's output, e.g., ['a', 'b', 'c', 'd', 'e'].
    """

    top_characters = []
    prediction = prediction[0]
    # Iterate over the 5 segments
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size
        
        # Get the index of the maximum value within this segment
        segment = prediction[start:end]
        top_index = np.argmax(segment)  # Find the index of the maximum value within the segment
        
        # Map the index to the corresponding character
        top_character = characters[top_index]
        top_characters.append(top_character)
    
    return top_characters


def load_images_and_labels(json_file):
    """
    Loads the images with their corresponding labels from the json file.

    Parameters:
    json_file (str): Path to the JSON file containing image paths and labels.

    Returns:
    tuple: A tuple containing:
        - images (list of np.ndarray): Processed image data.
        - labels (list of np.ndarray): Encoded labels corresponding to the images.
    """

    images = []
    labels = []
    
    # load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # iterate over the entries in the JSON file
    for entry in data:
        image_path = entry['image']

        # check if the image file exists
        if os.path.exists(image_path):
            # open and process the image
            img = Image.open(image_path).convert('L')   # convert to grayscale
            np_img = np.array(img)                      # convert to NumPy array

            # normalize the image to [0, 1]
            np_img = np_img / 255.0

            # append the processed image and its corresponding encoded label
            images.append(np_img)
            labels.append(encode_label(entry['label']))
    
    return images, labels