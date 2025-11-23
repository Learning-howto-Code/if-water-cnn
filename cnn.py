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
train_path= "/Users/abrahamhopkins/Downloads/Jakes_Model/still_images/still_images"
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