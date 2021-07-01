import os
import sys
import glob
import random
import shutil
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from train_binary_classification import create_res50

VAL_IMAGES_PATH = r'../../data/test'


def run_demo():
    model_name = "res50"
    model = create_res50(learning_rate=0.0001)
    params_folder = r'../../data/models'
    model.load_weights(os.path.join(params_folder, r'res50_1e-05_100.ckpt'))

    # results = model.evaluate()
    test_datagen = ImageDataGenerator(rescale= 1.0 / 255.)
    validation_generator = test_datagen.flow_from_directory(VAL_IMAGES_PATH, batch_size = 20, class_mode = 'binary', target_size = (512, 512))
    res = model.evaluate(validation_generator)
    print(res)


def run_evaluation():
    model_name = "res50"
    model = create_res50(learning_rate=0.0001)
    params_folder = r'../../data/models'
    model.load_weights(os.path.join(params_folder, r'res50_1e-05_100.ckpt'))

    # results = model.evaluate()
    test_datagen = ImageDataGenerator(rescale= 1.0 / 255.)
    validation_generator = test_datagen.flow_from_directory(VAL_IMAGES_PATH, batch_size = 20, class_mode = 'binary', target_size = (512, 512))
    res = model.evaluate(validation_generator)
    print(res)


if __name__ == "__main__":
    run_demo()