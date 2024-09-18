import numpy as np
from keras import models

import data


# ---------------------------------------------------------------------------------------------------------------------
# -- VARIABLES --------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
dir_small_margot = "testing/margot"
dir_small_jaime = "testing/jaime"

dir_margot = "margot_robbie"
dir_jaime = "jaime_pressly"

# load model
my_model = models.load_model("margot_or_jaime.keras")


# ---------------------------------------------------------------------------------------------------------------------
# -- FUNCTIONS --------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def evaluate_small_set(dir_margot, dir_jaime):
    # Load images and labels
    image_data, label_data = data.load_images_and_labels(dir_margot, dir_jaime)
    image_data = np.expand_dims(image_data, axis=-1)
    label_data = np.expand_dims(label_data, axis=-1)

    # Shuffle the data and labels together
    indices = np.arange(len(image_data))
    np.random.shuffle(indices)

    image_data = image_data[indices]
    label_data = label_data[indices]

    for img, lbl in zip(image_data, label_data):
        prediction = (my_model.predict(np.expand_dims(img, axis=0)))[0][0]
        
        print("\n--------------------------")
        if round(prediction) == 0:
            # Red or Green text based on correct/incorrect prediction
            if lbl == 0:
                print("\033[92mPredicted Margot Robbie with ", round((1 - prediction) * 100, 3), "% confidence.\033[0m")
            else:
                print("\033[91mPredicted Margot Robbie with ", round((1 - prediction) * 100, 3), "% confidence.\033[0m")
        else:
            # Red or Green text based on correct/incorrect prediction
            if lbl == 1:
                print("\033[92mPredicted Jaime Pressly with ", round(prediction * 100, 3), "% confidence.\033[0m")
            else:
                print("\033[91mPredicted Jaime Pressly with ", round(prediction * 100, 3), "% confidence.\033[0m")
        
        if lbl == 0:
            print("Ground truth: Margot Robbie.")
        else:
            print("Ground truth: Jaime Pressly.")


def evaluate(dir_margot, dir_jaime):
    # Load images and labels
    image_data, label_data = data.load_images_and_labels(dir_margot, dir_jaime)
    image_data = np.expand_dims(image_data, axis=-1)
    label_data = np.expand_dims(label_data, axis=-1)

    # Shuffle the data and labels together
    indices = np.arange(len(image_data))
    np.random.shuffle(indices)

    image_data = image_data[indices]
    label_data = label_data[indices]

    correct_counter = 0
    for img, lbl in zip(image_data, label_data):
        prediction = (my_model.predict(np.expand_dims(img, axis=0)))[0][0]

        # Print if prediction wrong
        if round(prediction) != lbl:
            predicted_name = "Margot Robbie" if round(prediction) == 0 else "Jaime Pressly"
            actual_name = "Margot Robbie" if lbl == 0 else "Jaime Pressly"
            
            # Red text for wrong predictions
            print("\033[91m\n--------------------------")
            print(f"Wrong prediction. Said {predicted_name}, but was {actual_name}.")
            print("\033[0m")  # Reset color to default
        else:
            correct_counter += 1
            # Green text for correct predictions
            print("\033[92mCorrect prediction!\033[0m")
    
    print("\n--------------------------")
    print(f"Correct predictions: {correct_counter} / {len(image_data)}, which is {100 * correct_counter / len(image_data)}%.")



evaluate_small_set(dir_small_margot, dir_small_jaime)
#evaluate(dir_margot, dir_jaime)