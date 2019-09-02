"""
responsible for extracting features from the preprocessed pictures in to a list of vectors.
saves vectors for later use.
"""



###################### Imports ###########################
import numpy
import cv2
import os
import string
import matplotlib.pyplot as plt
import time



###################### Constants ###########################
samples=[]
winSize = (32,32)
blockSize = (8,8)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64



def extractHogFeatures(path):
    """
    extract HoG features of images and save them in to lists.
    labels will be the name of the directory in which the images are placed.
    :return: list of all Hog Features.
    """


    feature_list = []
    label_list = os.listdir(path)
    if os.path.isdir(path):
        i = 0
        for subdir, dirs, files in os.walk(path):
            if i >= 3: continue
            i += 1
            letter_list = []
            for image in files:
                if str(image).endswith('.jpg'):
                    pathname = subdir + os.sep + image # true pathname of the image
                    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
                    im = cv2.imread(pathname)
                    h = hog.compute(im)
                    letter_list.append(h)
            if letter_list: # checks if list is empty
                feature_list.append(letter_list)
    return feature_list, label_list


