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
import matplotlib.pyplot as plt


VAL_IMAGES_PATH = r'../../data/test'


def run_demo():
    model_name = "res50"
    model = create_res50(learning_rate=0.0001)
    params_folder = r'../../data/models'
    model.load_weights(os.path.join(params_folder, r'res50_1e-05_100.ckpt'))

    # results = model.evaluate()
    test_datagen = ImageDataGenerator(rescale= 1.0 / 255.)
    # validation_generator = test_datagen.flow_from_directory(VAL_IMAGES_PATH, batch_size = 20, class_mode = 'binary', target_size = (512, 512))
    # res = model.evaluate(validation_generator)
    # print(res)
    validation_generator = test_datagen.flow_from_directory(VAL_IMAGES_PATH, batch_size = 20, class_mode = 'binary', target_size = (512, 512))
    img, label = validation_generator.next()

    predict = model.predict(img)
    print(predict)


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


def visualize_data_augmentation():
    datagen = ImageDataGenerator(rescale= 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    dir_It = datagen.flow_from_directory(VAL_IMAGES_PATH, batch_size=1, save_to_dir="output/", save_format='png')
    for _ in range(5):
        img, label = dir_It.next()
        print(img.shape)
        print("label is ", label)
        plt.imshow(img[0])
        plt.show()    




if __name__ == "__main__":
    # run_demo()
    # visualize_data_augmentation()
    run_demo()