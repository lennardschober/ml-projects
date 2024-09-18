from keras import layers, models


# CNN for CAPTCHA solving
def create_model(input_shape):
    """
    Creates a Convolutional Neural Network (CNN) model for image classification.

    The model applies image augmentation, convolutional layers for feature extraction
    and fully connected layers to classify whether Margot Robbie or Jaime Pressly are depicted.

    Args:
    - input_shape (tuple): Shape of the input images, expected to be (height, width, channels).

    Returns:
    - model (keras.Model): A compiled Keras Sequential model ready for training.
    
    Model Architecture:
    - Image Augmentation: Random rotation, translation, and contrast adjustment.
    - Convolutional Layers: Two Conv2D layers followed by MaxPooling and Dropout for feature extraction.
    - Dense Layers: Fully connected layers with Dropout for final classification.
    - Output: A Dense layer with 'sigmoid' activation for binary classification.
    """

    model = models.Sequential()

    # preprocessing layers for image augmentation
    model.add(layers.Input(shape=input_shape))
    model.add(layers.RandomFlip(mode="horizontal"))
    model.add(layers.RandomRotation(factor=(-0.15, 0.15)))
    model.add(layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)))
    model.add(layers.RandomContrast(factor=0.15))
    
    # CNN layers for feature extraction
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # flatten for output
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model