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
    features  = np.array(feature_list)
    labels = np.array(labels)[0:2]

    # data must be reshaped:
    features = features.reshape(1,) # should drop the single dimensions of features. should now be of 62X1024X144 array.
    features = features.reshape(1, 16, 9)  # drops array from 2d to 1d.
    new_labels = []
    x = 0
    for i in labels:
        new_labels.extend([i for j in range(len(features[x]))])
        x += 1

    X_train, X_test, y_train, y_test = train_test_split(features, new_labels)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # get the model accuracy
    model_score = knn.score(X_test, y_test)

    # save trained model
    joblib.dump(knn, '/models/knn_model.pkl')



KNN_model(pathname)
