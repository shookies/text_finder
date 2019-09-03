"""
takes a number of ML models and saves them for the classification of letters
in to actual characters
"""


###################### Imports ###########################
import numpy as np
import cv2
import os
import string
import matplotlib.pyplot as plt
import extractFeatures
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


###################### Constants ###########################
pathname = '/cs/usr/punims/Desktop/OCRProject/Preprocessed_Images'
NUM_OF_IMAGES = 0
NUM_OF_BINS = 9






###################### Functions ###########################


def KNN_model(path):
    """
    creates KNN model from path of files, extracting the HoG features and creating the model
    :return:
    """

    feature_list, labels = extractFeatures.extractHogFeatures(path)
    features = np.array(feature_list)
    labels_ascii = list(map(ord, labels))  #turns string labels in to ascii integer values.
    labels_ascii = np.array(labels_ascii)


    X_train, X_test, y_train, y_test = train_test_split(features, labels_ascii)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # get the model accuracy
    model_score = knn.score(X_test, y_test)

    # save trained model
    joblib.dump(knn, '/models/knn_model.pkl')



KNN_model(pathname)
