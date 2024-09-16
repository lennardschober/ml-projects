from keras import layers, models


# global attention layer
class GlobalAttention(layers.Layer):
    """
    A custom Keras layer that applies global attention to focus on important regions
    in the input feature maps.

    The attention mechanism uses a 1x1 convolution to generate attention weights 
    for each spatial location, normalizes them with batch normalization, and applies
    a sigmoid activation to constrain the weights between 0 and 1. The resulting
    weights are used to modulate the input feature maps by element-wise multiplication.

    Args:
    - num_channels (int): Number of channels in the input feature maps (e.g., 64).

    Methods:
    - call(x, training=False): Applies the attention mechanism to the input feature maps.
    """

    def __init__(self, num_channels, **kwargs):
        """
        Initializes the GlobalAttention layer.

        Args:
        - num_channels (int): Number of channels in the input feature maps.
        - kwargs: Additional keyword arguments for the Keras Layer class.
        """

        super(GlobalAttention, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.attention_conv = layers.Conv2D(1, kernel_size=1, padding='same', use_bias=False)
        self.batch_norm = layers.BatchNormalization()
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x, training=False):
        """
        Applies the attention mechanism to the input feature maps.

        Args:
        - x (tensor): Input tensor, expected shape (batch_size, height, width, num_channels).
        - training (bool): Flag indicating whether the layer should behave in training mode or inference mode.

        Returns:
        - tensor: The input feature maps modulated by the attention weights.
        """
        
        attention_weights = self.attention_conv(x)
        attention_weights = self.batch_norm(attention_weights, training=training)
        attention_weights = self.sigmoid(attention_weights)
        return x * attention_weights


# CNN for CAPTCHA solving
def create_model(input_shape, num_classes):
    """
    Creates a Convolutional Neural Network (CNN) model for captcha recognition.

    The model applies image augmentation, convolutional layers for feature extraction, 
    a custom attention mechanism, and fully connected layers to predict the sequence 
    of characters in a captcha image.

    Args:
    - input_shape (tuple): Shape of the input images, expected to be (height, width, channels).
    - num_classes (int): The number of possible character classes (e.g., alphanumeric characters).

    Returns:
    - model (keras.Model): A compiled Keras Sequential model ready for training.
    
    Model Architecture:
    - Image Augmentation: Random rotation, translation, and contrast adjustment.
    - Convolutional Layers: Two Conv2D layers followed by BatchNormalization, MaxPooling, and Dropout for feature extraction.
    - Attention Mechanism: Custom GlobalAttention layer to focus on important features.
    - Dense Layers: Fully connected layers with Dropout for final classification.
    - Output: A Dense layer with 'softmax' activation for predicting 5 characters.
    """

    model = models.Sequential()

    # preprocessing layers for image augmentation
    model.add(layers.Input(shape=input_shape))
    model.add(layers.RandomRotation(factor=(-0.02, 0.02)))
    model.add(layers.RandomTranslation(height_factor=(-0.02, 0.02), width_factor=(-0.02, 0.02)))
    model.add(layers.RandomContrast(factor=0.02))
    
    # CNN layers for feature extraction
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # global attention
    model.add(GlobalAttention(num_channels=64))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    # Flatten for output
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    # output
    model.add(layers.Dense(5 * num_classes, activation='softmax'))
    
    return model