import os
import sys
import cv2 
from skimage.util import random_noise
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt


def create_noise_data(src_folder, dst_folder, noise_type="Gaussian"):
    """add noise to all images in the input_folder

    Args:
        src_folder ([str]): path to the source images
        dst_folder ([str]): save path
        noise_type (str, optional): ['Gaussian', 'S&P']. Defaults to "Gaussian".
    """
    if not os.path.exists(src_folder):
        os.makedirs(src_folder)
    
    ## remove the target folder if it is exist
    dst_folder = os.path.join(dst_folder, noise_type)
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder, ignore_errors=True)

    ## copy the origin data
    shutil.copytree(src_folder, dst_folder)
    create_noise_image_recursively(dst_folder, noise_type=noise_type)


def create_noise_image_recursively(input_folder, noise_type="Gaussian"):
    """add noise to all images in the input_folder

    Args:
        input_folder ([str]): path to the image folder
        noise_type (str, optional): ['Gaussian', 'S&P']. Defaults to "Gaussian".
    """
    # print(input_folder)
    root_path = next(os.walk(input_folder))[0]
    images = next(os.walk(input_folder))[2]
    for image in images:
        try:
            img = plt.imread(os.path.join(root_path, image))
            if img is not None:
                if noise_type == "Gaussian":
                    noise_img = random_noise(img, mode="gaussian", mean=0, var=0.05, clip=True)
                elif noise_type == "S&P":
                    noise_img = random_noise(img, mode="s&p", salt_vs_pepper=0.5, clip=True)
                else:
                    raise TypeError("Unknown noise type")
                plt.imsave(os.path.join(root_path, image), noise_img)
        except:
            print(os.path.join(root_path, image) + " is not an image")
    for subdir in next(os.walk(input_folder))[1]:
        create_noise_image_recursively(os.path.join(input_folder, subdir), noise_type)



if __name__ == "__main__":
    src_folder = r'D:\workspace\my_git_repos\image_feature_detection\data\real_test_data'
    dst_folder = r'D:\workspace\my_git_repos\image_feature_detection\tmp'
    noise_type = "S&P"
    create_noise_data(src_folder, dst_folder, noise_type)