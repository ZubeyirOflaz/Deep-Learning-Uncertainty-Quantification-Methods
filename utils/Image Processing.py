import cv2
import numpy as np
import os
import logging
import random
from  pathlib import Path

'''This script is used in order to apply motion and gaussian blur to the casting dataset in order. The last function,
create_blur_database receives the directory of images, the desired ratio of motion blurred images, motion blur kernel 
size and gaussian blur sigma level. It then randomly divides images and applies motion and gaussian blur to all the 
images in a given directors and saves it to a new folder 
'''


def apply_motion_blur(image_dir, final_dir, kernel_size=15):
    img = cv2.imread(image_dir)
    file_path, file_extension = os.path.splitext(image_dir)
    file_name = Path(image_dir).stem
    # Specify the kernel size.
    # The greater the size, the more the motion.
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    motion_blurred = cv2.filter2D(img, -1, kernel)
    file_destination = final_dir + '\\' + file_name + '_mb' + file_extension
    cv2.imwrite(file_destination, motion_blurred)
    logging.debug(f'{file_name} has been modified with motion blur')


def apply_gaussian_blur(image_dir,final_dir, filter_sigma=2):
    img = cv2.imread(image_dir)
    file_path, file_extension = os.path.splitext(image_dir)
    file_name = Path(image_dir).stem
    '''cv2.imshow('image', img);
    cv2.waitKey(0);
    cv2.destroyAllWindows();
    cv2.waitKey(1)
'''
    blurred_image = cv2.GaussianBlur(img, (15,15), 2)
    file_destination = final_dir + '\\' + file_name + '_gaussian' + file_extension
    if not cv2.imwrite(file_destination, blurred_image):
        logging.error('Could not write the image')
    logging.debug(f'{file_name} has been modified with gaussian blur')


def create_blur_database(directory_location,final_dir, mb_gaussian_ratio=0.75):
    image_files = os.listdir(directory_location)
    image_files = [directory_location + '\\' + x for x in image_files]
    random.shuffle(image_files)
    separation_point = int(len(image_files) * mb_gaussian_ratio)
    logging.debug(
        f'{str(separation_point)} will be applied motion blur and {str((len(image_files) - separation_point))} will '
        f'be applied gaussian blur')
    motion_blur_list = image_files[:separation_point]
    gaussian_list = image_files[separation_point:]
    for i in motion_blur_list:
        apply_motion_blur(i, final_dir)
    for i in gaussian_list:
        apply_gaussian_blur(i, final_dir)
    print(len(gaussian_list))
    print(separation_point)
    print(len(image_files))

casting_train_ok = r'N:\Thesis\Datasets\casting_dataset\casting_data\casting_data\test\ok_front'
final_dir = r'N:\Thesis\modified\test\ok_front'
create_blur_database(casting_train_ok, final_dir)

