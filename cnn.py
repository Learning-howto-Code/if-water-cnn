# ignore errors in the imports
#SSH at (venv) Abrahams-MacBook-Pro:if_water abrahamhopkins$ 
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
train_path= "//Users/abrahamhopkins/Downloads/Jakes_Model/if_water/train"
val_path= "/Users/abrahamhopkins/Downloads/Jakes_Model/if_water/val"


datagen= ImageDataGenerator(rescale=1./255)
#prepares imgs for training
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

def stack_frames_generator(base_gen, frames_per_sample=5):
    while True:
        # Fetch frames_per_sample * batch_size images from the base generator
        imgs = []
        labels = []
        for _ in range(frames_per_sample):
            x, y = next(base_gen)
            imgs.append(x)
            labels.append(y)

        # x has shape (batch_size, H, W, 3)
        # We want to stack along channel axis per item
        imgs = np.stack(imgs, axis=1)  # (batch_size, 5, H, W, 3)

        # Merge the frame and channel dimensions → (batch_size, H, W, 5*3)
        b, f, h, w, c = imgs.shape
        imgs = imgs.reshape(b, h, w, f*c)

        # You MUST ensure labels match. If all 5 frames belong to same class, use labels[0]
        labels = labels[0]

        yield imgs, labels
stacked_train = stack_frames_generator(train_data, frames_per_sample=5)
stacked_valid = stack_frames_generator(valid_data, frames_per_sample=5)

# CNN Layers
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

# Trains the model

model.fit(
    stacked_train,
    steps_per_epoch=train_data.samples // batch_size,
    validation_data=stacked_valid,
    validation_steps=valid_data.samples // batch_size,
    epochs=5
)

model.save("model.keras")