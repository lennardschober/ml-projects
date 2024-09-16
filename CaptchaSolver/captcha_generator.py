import os
import json

import random
from tqdm import tqdm
from io import BytesIO
from captcha.image import ImageCaptcha

import helper


# ---------------------------------------------------------------------------------------------------------------------
# -- GLOBAL VARIABLES -------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
input_dir = helper.input_dir                # directory of training images
annotations_file = helper.annotations_file  # file containing annotations
sequence_length = helper.sequence_length    # output sequence length
characters = helper.characters              # character set
num_files = 500                             # number of captchas that will be created

# initialize captcha object
captcha = ImageCaptcha(fonts=["arial.ttf"])
captcha.character_offset_dx = (0, 0)
captcha.character_offset_dy = (0, 0)
captcha.character_rotate = (-25, 25)
captcha.character_warp_dx = (-0.1, 0.1)
captcha.character_warp_dy = (-0.1, 0.1)
captcha.word_space_probability = 0
captcha.word_offset_dx = 0.1


# ---------------------------------------------------------------------------------------------------------------------
# -- CREATE CAPTCHAS AND ANNOTATIONS ----------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def create_annotations(image_folder, output_file):
    """
    Generates a JSON file containing image paths and corresponding labels for CAPTCHA images.

    This function reads `.png` image files from a given folder, finds their corresponding `.txt` label 
    files, and extracts the label (a sequence of 5 character values). It then creates a list of 
    annotations, where each entry contains the path to the image and its associated label. The 
    annotations are saved as a JSON file.

    Args:
    - image_folder (str): The directory containing the CAPTCHA image files and their respective `.txt` label files.
    - output_file (str): The path where the generated JSON file with annotations will be saved.

    Returns:
    - None: The function writes the annotations directly to a JSON file.

    Notes:
    - Each `.png` file in the `image_folder` should have a corresponding `.txt` file containing 
      a comma-separated list of 5 label values.
    - If any `.png` file is missing its corresponding `.txt` file, or if the label file has an invalid 
      number of values, the function will skip that image.

    Example:
    - The JSON output will have entries in the following format:
      {
          "image": "path/to/image.png",
          "label": ["char1", "char2", "char3", "char4", "char5"]
      }
    """

    annotations = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            # Extract background name from filename
            captcha = os.path.splitext(filename)[0]
            
            # Read bounding box coordinates from the corresponding .txt file
            label_file = os.path.join(image_folder, f"{captcha}.png.txt")
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    line = f.readline().strip()
                    # Split the line into 5 bounding box x-values
                    label_values = [x.strip() for x in line.split(',')]
                    
                    if len(label_values) != 5:
                        print(f"Invalid number of values for {filename}")
                        continue

            else:
                print(f"Label file missing for {filename}")
                continue

            # Append to annotations list
            annotations.append({
                'image': os.path.join(image_folder, filename),
                'label': label_values
            })
    
    # Write annotations to JSON file
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)


def create_captchas(p_num_files=num_files, p_length=sequence_length):
    """
    Generates CAPTCHA images and corresponding labels, saving them to the specified directory.

    This function creates a specified number of CAPTCHA images with randomly generated strings of a
    given length. It writes the images to disk as `.png` files and stores their corresponding labels
    in `.txt` files. Each label is a sequence of characters used in the CAPTCHA image.

    Args:
    - p_num_files (int, optional): The number of CAPTCHA files to generate. Defaults to `num_files`.
    - p_length (int, optional): The length of the random string in each CAPTCHA. Defaults to `sequence_length`.

    Returns:
    - None: The function writes the generated files directly to the disk.

    Example:
    - For each CAPTCHA image saved as 'i.png', a corresponding label will be saved in 'i.png.txt'.
    """

    for i in tqdm(range(p_num_files), desc="Creating files."):
        random_string = ''.join(random.choices(characters, k=p_length))
        data: BytesIO = captcha.generate(random_string)
        # save captcha image
        captcha.write(random_string, f'{input_dir}/{i}.png')
        # save captcha label
        with open(os.path.join(input_dir, f"{i}.png.txt"), 'w') as f:
            f.write(", ".join(map(str, random_string)) + "\n")


create_captchas()
create_annotations(input_dir, annotations_file)