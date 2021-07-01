import os
import sys
import glob
import random
import shutil
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
from tensorflow.keras.applications import ResNet50, EfficientNetB7
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop


IMAGES_FOLDER = r'../../data/images'
TRAIN_IMAGES_PATH = r'../../data/train'
VAL_IMAGES_PATH = r'../../data/test'

def split_train_test_data(test_size=0.2):
    class_ids = next(os.walk(IMAGES_FOLDER))[1]
    for class_id in class_ids:
        images = glob.glob(os.path.join(IMAGES_FOLDER, class_id, '*'))
        n_images = len(images)
        n_train_images = int((1 - test_size) * n_images)
        n_val_images = n_images - n_train_images
        train_images = random.sample(images, n_train_images)
        val_images = random.sample(images, n_val_images)
        train_dst_folder = os.path.join(TRAIN_IMAGES_PATH, class_id)
        if not os.path.exists(train_dst_folder):
            os.makedirs(train_dst_folder)
        for image in train_images:
            shutil.copy(image, train_dst_folder)
        val_dst_folder = os.path.join(VAL_IMAGES_PATH, class_id)
        if not os.path.exists(val_dst_folder):
            os.makedirs(val_dst_folder)
        for image in val_images:
            shutil.copy(image, val_dst_folder)


def create_res50(learning_rate):
    final_model = Sequential()
    final_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    final_model.add(Dense(1, activation='sigmoid'))
    final_model.compile(optimizer = tf.keras.optimizers.SGD(lr=learning_rate), loss = 'binary_crossentropy', metrics = ['acc'])
    return final_model


def train_res50():
    ## Data Augmentation 
    train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1.0/255.)

    train_generator = train_datagen.flow_from_directory(TRAIN_IMAGES_PATH, batch_size = 20, class_mode = 'binary', target_size = (512, 512))

    validation_generator = test_datagen.flow_from_directory(VAL_IMAGES_PATH, batch_size = 20, class_mode = 'binary', target_size = (512, 512))
    model_name = 'res50'
    learning_rate = 0.0001
    epochs = 100
    ## import base model
    # base_model = ResNet50(input_shape=(512, 512, 3), include_top=False, weights="imagenet")
    # for layer in base_model.layers:
    #     layer.trainable = False

    # x = layers.Flatten()(base_model.output)
    # x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dropout(0.3)(x)
    # x = layers.Dense(1, activation='sigmoid')(x)
    # final_model = tf.keras.models.Model(base_model.input, x)

    # final_model.compile(optimizer = RMSprop(lr=learning_rate), loss = 'binary_crossentropy', metrics = ['acc'])
    final_model = create_res50(learning_rate)
    
    ## train model
    resnet_history = final_model.fit(train_generator, validation_data = validation_generator, epochs = epochs)

    ## save model
    save_path = r'data/models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    final_model.save_weights(os.path.join(save_path, '_'.join([model_name, str(learning_rate), str(epochs)]) + '.ckpt'))



def train_efficientnetB7():

    train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1.0/255.)

    train_generator = train_datagen.flow_from_directory(TRAIN_IMAGES_PATH, batch_size = 20, class_mode = 'binary', target_size = (512, 512))

    validation_generator = test_datagen.flow_from_directory(VAL_IMAGES_PATH, batch_size = 20, class_mode = 'binary', target_size = (512, 512))
    model_name = 'EfficientNetB7'
    learning_rate = 0.0001
    epochs = 100
    base_model = EfficientNetB7(input_shape=(512, 512, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model_final = Model(base_model.input, predictions)
    model_final.compile(optimizer = RMSprop(lr=learning_rate) ,loss='binary_crossentropy',metrics=['accuracy'])
    eff_history = model_final.fit_generator(train_generator, validation_data = validation_generator, epochs = epochs)
    save_path = r'data/models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_final.save_weights(os.path.join(save_path, '_'.join([model_name, str(learning_rate), str(epochs)]) + '.ckpt'))


if __name__ == "__main__":
    # split_train_test_data()
    # train_res50()
    train_efficientnetB7()