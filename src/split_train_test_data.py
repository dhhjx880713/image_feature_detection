import os
import glob
import random
import shutil


IMAGES_FOLDER = r'../data/images'
TRAIN_IMAGES_PATH = r'../data/train_big_negative_set'
VAL_IMAGES_PATH = r'../data/test_big_negative_set'


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
        

if __name__ == "__main__":
    split_train_test_data()