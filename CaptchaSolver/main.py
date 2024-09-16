import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers, losses, callbacks

import model
import helper


# ---------------------------------------------------------------------------------------------------------------------
# -- GLOBAL VARIABLES -------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
annotations_file = helper.annotations_file  # file containing annotations
characters = helper.characters              # character set


# ---------------------------------------------------------------------------------------------------------------------
# -- DATA PREPARATION -------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# Convert images and labels to NumPy arrays with additional (batch) dimension
image_data, label_data= helper.load_images_and_labels(annotations_file)
image_data = np.expand_dims(image_data, axis=-1)
label_data = np.expand_dims(label_data, axis=-1)


# ---------------------------------------------------------------------------------------------------------------------
# -- TRAINING ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
input_shape = (60, 160, 1)          # dimensions of captchas
num_classes = len(characters)       # number of possible characters (1-9, a-z)
batch_size = 256                    # batch size for training
num_epochs = 5000                   # number of epochs for training
size_training_set = len(image_data) # number of images in the training set
val_split = 0.2                     # portion of training set that is used for validation

# create the model
my_model = model.create_model(input_shape, num_classes)

# custom learning rate scheduler
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,                                                 # initial learning rate of 0.0001
    decay_steps=int(size_training_set * (1 - val_split) / batch_size) * 200,    # every 200 epochs
    decay_rate=0.95                                                             # after decay_steps, update learning
)                                                                               # rate: 0.0001 * 0.95^(#decay_steps)

# custom optimizer
opt = optimizers.Adam(
    learning_rate=lr_schedule,
    clipnorm=1.0
)

# compile and display model
my_model.compile(
    optimizer='adam',
    loss=losses.CategoricalFocalCrossentropy()
)
my_model.summary()

# custom criterium for early stopping:
# stop training if validation loss didn't decrease for 200 epochs, ignoring the first 200 epochs
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=200,
    restore_best_weights=True,
    start_from_epoch=200
)

# start training and store data in history variable
history = my_model.fit(
        image_data, label_data, 
        epochs=num_epochs, 
        batch_size=batch_size,
        validation_split=val_split, 
        shuffle=True, 
        callbacks=[early_stopping]
    )


# ---------------------------------------------------------------------------------------------------------------------
# -- SAVING & VISUALIZATION -------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
my_model.save("captcha_solver.keras")

# access loss and validation loss from the history object
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting loss
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()