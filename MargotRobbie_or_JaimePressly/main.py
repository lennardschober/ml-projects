import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers, losses, callbacks

import data
import model

# ---------------------------------------------------------------------------------------------------------------------
# -- VARIABLES --------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
dir_margot = "margot_robbie"
dir_jaime = "jaime_pressly"

# ---------------------------------------------------------------------------------------------------------------------
# -- DATA PREPARATION -------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# Convert images and labels to NumPy arrays with additional (batch) dimension
image_data, label_data = data.load_images_and_labels(dir_margot, dir_jaime)
image_data = np.expand_dims(image_data, axis=-1)
label_data = np.expand_dims(label_data, axis=-1)


# ---------------------------------------------------------------------------------------------------------------------
# -- TRAINING ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
input_shape = (355, 218, 1)             # dimensions of images
batch_size = 32                         # batch size for training
num_epochs = 1000                       # number of epochs for training
size_training_set = len(image_data)     # number of images in the training set
val_split = 0.2                         # portion of training set that is used for validation
class_weight = {0: 443 / 391, 1: 1.0}   # Adjust the weights based on the class distribution


my_model = model.create_model(input_shape)

# custom learning rate scheduler
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,                                             # initial learning rate of 0.001
    decay_steps=int(size_training_set * (1 - val_split) / batch_size) * 50, # every 50 epochs
    decay_rate=0.75                                                         # after decay_steps, update learning
)                                                                           # rate: 0.001 * 0.75^(#decay_steps)

# custom optimizer
opt = optimizers.Adam(
    learning_rate=lr_schedule
)

# compile and display model
my_model.compile(
    optimizer='adam',
    metrics=['acc'],
    loss=losses.BinaryCrossentropy()
)
my_model.summary()

# custom criterium for early stopping:
# stop training if validation loss didn't decrease for 100 epochs, ignoring the first 200 epochs
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=100,
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
        callbacks=[early_stopping],
        class_weight=class_weight
    )


# ---------------------------------------------------------------------------------------------------------------------
# -- SAVING & VISUALIZATION -------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
my_model.save("margot_or_jaime.keras")

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
