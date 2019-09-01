"""
Preprocess for chars74k dataset of images
"""

###################### Imports ###########################
import pytesseract as pt
import numpy as np
from pytesseract import Output
import cv2
from os import listdir
import os.path
###################### Functions ###########################

def open_image(filename, scale_to=[32, 32]):
    """Opens an image, returns the preprocessed image (scaled, masked)"""
    img = cv2.imread(filename) * cv2.imread(filename.replace('Bmp', 'Msk'))/255
    processed_img = np.zeros(list(scale_to)+[3])
    img = cv2.resize(img, tuple(scale_to))
    # normalising
    processed_img = img.astype(np.float32)
    for c in range(3):
        processed_img[:,:,c] /= np.max(processed_img[:,:,c])
    # to grayscale
    processed_img = cv2.cvtColor(
        (processed_img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    processed_img = np.expand_dims(processed_img, -1)

    return processed_img

def preprocess_images(file_path, new_file_path):
    """
    saves images after preprocessing them
    :param file_path:
    :param new_file_path:
    :return:
    """
    if not os.path.isdir(new_file_path):
        os.mkdir(new_file_path)
    i = 0
    for dir in listdir(file_path):
        j = 0
        for image_path in listdir(file_path + '/' + dir):
            image = open_image(image_path)
            cv2.imwrite(file_path + '/' + image_path + '/' str(i) + '/' +str(i) + '.jpg', image)
            j += 1
        i += 1

