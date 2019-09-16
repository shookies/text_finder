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
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, Add, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


###################### Constants ###########################
pathname = '/cs/usr/punims/Desktop/OCRProject/Preprocessed_Images'
saved_models_path = '/cs/usr/punims/Desktop/OCRProject/MLmodels/knn_2n.gz'
path_to_filename = '/cs/usr/punims/Desktop/OCRProject/MLmodels/cnn1'
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
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)

    # get the model accuracy
    model_score = knn.score(X_test, y_test)
    print("the model score is: " + str(model_score))

    # save trained model
    joblib.dump(knn, saved_models_path)


def CNN_Model(path):
    """
    a CNN for recognizing letters from the database within the path
    :param path: based on architecture from https://www.ijraset.com/fileserve.php?FID=16040
    :return:
    """

    # Get images and labels
    feature_list, labels = extractFeatures.imagesAsIs(path)
    features = np.array(feature_list)
    labels_ascii = list(map(ord, labels))  #turns string labels in to ascii integer values.
    labels_ascii = np.array(labels_ascii)

    # split in to test and train datasets
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels_ascii)

    # CNN architecture
    input = Input(shape=(32, 32, 1))
    conv1 = Conv2D(32, (5, 5), padding='same')(input)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(48, (5, 5), padding='same')(pool1)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, (5, 5), padding='same')(pool2)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    output = Dense(62)(pool3)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train , Y_train , batch_size=100, epochs=5 , validation_split=0.2)  # TODO doesn't work, check why

    # Save the model
    model.save_weights(path_to_filename)


# KNN_model(pathname)
CNN_Model(pathname)
