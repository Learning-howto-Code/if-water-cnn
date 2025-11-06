import numpy as np
import matplotlib as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_to_folder = "/Users/jakehopkins/Downloads/if_water"
# Filepaths
train_path= "/Users/abrahamhopkins/Downloads/Jakes_Model/still_images/still_images"
val_path= "/Users/abrahamhopkins/Downloads/Jakes_Model/if_water/val"


datagen= ImageDataGenerator(rescale=1./255)
# Data generators
batch_size = 32
image_size = (224, 224)
class_mode = 'binary'

train_data = datagen.flow_from_directory(
    train_path,
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    seed=42
)

valid_data = datagen.flow_from_directory(
    val_path,
    batch_size=batch_size,
    target_size=image_size,
    class_mode=class_mode,
    seed=42
)

# Create a CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(10, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(10, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(10, (3, 3), activation="relu"),
    tf.keras.layers.Conv2D(10, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

# Fit the model
epochs = 5
history_1 = model.fit(train_data, epochs=epochs, validation_data=valid_data, validation_steps=len(valid_data))
