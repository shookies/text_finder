"""
responsible for extracting features from the preprocessed pictures in to a list of vectors.
saves vectors for later use.
"""



###################### Imports ###########################
import numpy as np
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
    label_list = []
    labels = os.listdir(path)
    if os.path.isdir(path):
        # i = 0  #TODO delete this
        cur_letter = -1  #done for this specific file where it is saved.
        for subdir, dirs, files in os.walk(path):
            if len(files) == 0:
                continue
            # if i >= 3: continue #TODO delete this
            # i += 1 #TODO delete this
            cur_letter += 1
            letter_list = []
            for image in files:
                label_list.append(labels[cur_letter])
                if str(image).endswith('.jpg'):
                    pathname = subdir + os.sep + image # true pathname of the image
                    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
                    im = cv2.imread(pathname)
                    h = np.array(hog.compute(im)).squeeze()
                    letter_list.append(h)
            if len(feature_list) == 0: # checks if list is empty
                feature_list = np.array(letter_list)
            else:
                feature_list = np.concatenate((feature_list, np.array(letter_list)))
    return feature_list, label_list


