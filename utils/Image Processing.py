import cv2
import numpy as np
import os
import logging
from skimage.filters import gaussian
import random

'''This script is used in order to apply motion and gaussian blur to the casting dataset in order. The last function,
create_blur_database receives the directory of images, the desired ratio of motion blurred images, motion blur kernel 
size and gaussian blur sigma level. It then randomly divides images and applies motion and gaussian blur to all the 
images in a given directors and saves it to a new folder 
'''


def apply_motion_blur(image_dir, kernel_size):
    img = cv2.imread(image_dir)
    file_name, file_extension = os.path.splitext(image_dir)
    # Specify the kernel size.
    # The greater the size, the more the motion.
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    motion_blurred = cv2.filter2D(img, -1, kernel)
    file_destination = 'modified/' + file_name + '_mb' + file_extension
    cv2.imwrite(file_destination, motion_blurred)
    logging.debug(f'{file_name} has been modified with motion blur')


def apply_gaussian_blur(image_dir, filter_sigma):
    img = cv2.imread(image_dir)
    file_name, file_extension = os.path.splitext(image_dir)
    # Specify the kernel size.
    # The greater the size, the more the motion.
    blurred_image = gaussian(img, sigma=(filter_sigma, filter_sigma))
    file_destination = 'modified/' + file_name + '_gaussian' + file_extension
    cv2.imwrite(file_destination, blurred_image)
    logging.debug(f'{file_name} has been modified with gaussian blur')


def create_blur_database(directory_location, mb_gaussian_ratio=0.9, mb_kernel=20, gaussian_sigma=1):
    image_files = os.listdir(directory_location)
    random.shuffle(image_files)
    separation_point = int(len(image_files) / mb_gaussian_ratio)
    logging.debug(
        f'{str(separation_point)} will be applied motion blur and {str((len(image_files) - separation_point))} will '
        f'be applied gaussian blur')
    motion_blur_list = image_files[:separation_point]
    gaussian_list = image_files[separation_point:]
    map(apply_motion_blur, motion_blur_list)
    map(apply_gaussian_blur, gaussian_list)
