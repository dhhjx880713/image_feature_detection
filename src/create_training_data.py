from ntpath import join
import os
import sys
import glob
import shutil
import matplotlib.pyplot as plt
import random


def create_training_data_for_binary_classification():
    """select images randomly from imagenet as negative examples for binary classification, the number of images is equal to the size of architecture dataset
       A subset of ImageNet named  
    """
    source_image_folder = r'../data/imagenette2/train'
    hui_architecture_data_folder = r'../data/images/0'
    dst_folder = r'../data/images/1'
    ## remove the destination folder if it exists
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder, ignore_errors=True)
    os.makedirs(dst_folder)
    n_images = len(glob.glob(os.path.join(hui_architecture_data_folder, '*')))
    subfolders = next(os.walk(source_image_folder))[1]
    n_classes = len(subfolders)
    n_samples_per_class = n_images // n_classes

    for subfolder in subfolders[:-1]:
        images_in_subfolder = glob.glob(os.path.join(source_image_folder, subfolder, '*.jpeg'))
        if len(images_in_subfolder) > n_samples_per_class:
            sampled_images = random.sample(images_in_subfolder, n_samples_per_class)
        else:
            sampled_images = images_in_subfolder
        for image in sampled_images:
            shutil.copy(image, dst_folder)
    res = n_images - (n_classes - 1) * n_samples_per_class
    images_in_last_folder = glob.glob(os.path.join(source_image_folder, subfolders[-1], '*.jpeg'))
    if len(images_in_last_folder) > res:
        sampled_images = random.sample(images_in_last_folder, res)
    else:
        sampled_images = images_in_last_folder
        print("cannot sample enough images")
    for image in sampled_images:
        shutil.copy(image, dst_folder)




if __name__ == "__main__":
    create_training_data_for_binary_classification()
