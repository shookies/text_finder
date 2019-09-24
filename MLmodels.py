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
from sklearn.externals import joblib
import extractFeatures
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, Add, MaxPool2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


###################### Constants ###########################
pathname = '/cs/usr/punims/Desktop/OCRProject/Preprocessed_Images'
saved_models_path = '/cs/usr/punims/Desktop/OCRProject/MLmodels/knn_2n.gz'
path_to_filename = '/cs/usr/punims/Desktop/OCRProject/MLmodels/cnn1'
path_of_training_data = '/cs/usr/punims/Desktop/OCRProject/saved_data_sets/train_data.pkl'
path_of_training_labels ='/cs/usr/punims/Desktop/OCRProject/saved_data_sets/train_labels.pkl'
path_of_test_data = '/cs/usr/punims/Desktop/OCRProject/saved_data_sets/test_data.pkl'
path_of_test_labels = '/cs/usr/punims/Desktop/OCRProject/saved_data_sets/test_labels.pkl'

cnn_training_data_path = '/cs/usr/punims/Desktop/OCRProject/cnn_saved_data_sets/train_data.pkl'
cnn_test_data_path = '/cs/usr/punims/Desktop/OCRProject/cnn_saved_data_sets/test_data.pkl'
cnn_training_labels_path = '/cs/usr/punims/Desktop/OCRProject/cnn_saved_data_sets/train_labels.pkl'
cnn_test_labels_path = '/cs/usr/punims/Desktop/OCRProject/cnn_saved_data_sets/test_labels.pkl'
cnn_model_path = '/cs/usr/punims/Desktop/OCRProject/cnn_saved_data_sets/cnn_model.gz'

NUM_OF_IMAGES = 0
NUM_OF_BINS = 9






###################### Functions ###########################


def KNN_model(path):
    """
    creates KNN model from path of files, extracting the HoG features and creating the model
    :return:
    """

    if not os.path.isfile(path_of_test_data):

        # extract hog features, append labels,
        feature_list, labels = extractFeatures.extractHogFeatures(path)
        features = np.array(feature_list)

        labels_ascii = list(map(ord, labels))  #turns string labels in to ascii integer values.
        labels_ascii = np.array(labels_ascii)

        # split the data
        X_train, X_test, y_train, y_test = train_test_split(features, labels_ascii)

        # create save files for the data
        joblib.dump(X_train, path_of_training_data)
        joblib.dump(y_train, path_of_training_labels)
        joblib.dump(X_test, path_of_test_data)
        joblib.dump(y_test, path_of_training_labels)

    else:
        X_train = joblib.load(path_of_training_data)
        X_test = joblib.load(path_of_test_data)
        y_train = joblib.load(path_of_training_labels)
        y_test = joblib.load(path_of_test_labels)

    # create model
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)

    # get the model accuracy
    model_score = knn.score(X_test, y_test)
    print("the model score is: " + str(model_score))

    # save trained model
    joblib.dump(knn, saved_models_path)


def CNN_Model(path, override_dataset):
    """
    a CNN for recognizing letters from the database within the path
    :param path: based on architecture from https://www.ijraset.com/fileserve.php?FID=16040
    :return:
    """

    if override_dataset:

        # Get images and labels
        feature_list, labels = extractFeatures.imagesAsIs(path)
        features = np.array(feature_list)
        labels_ascii = list(map(ord, labels))  #turns string labels in to ascii integer values.
        labels_ascii = np.array(labels_ascii)

    # split in to test and train datasets

        # X_train, X_test, y_train, y_test = train_test_split(features, labels_ascii)
        X = features
        y = labels_ascii
        X = X.reshape(-1, 32, 32, 1)
        # X_test = X_test.reshape(-1, 32, 32, 1)

        # create save files for the data
        joblib.dump(X, cnn_training_data_path)
        joblib.dump(y, cnn_training_labels_path)
        # joblib.dump(X_test, cnn_test_data_path)
        # joblib.dump(y_test, cnn_test_labels_path)

    else:
        X = joblib.load(cnn_training_data_path)
        y = joblib.load(cnn_training_labels_path)

    # CNN architecture
    input = Input(shape=(32, 32, 1))
    conv1 = Conv2D(32, (5, 5), padding='same', activation='relu')(input)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(48, (5, 5), padding='same', activation='relu')(pool1)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(62, (5, 5), padding='same', activation='relu')(pool2)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    flatten = Flatten()(pool3)
    fully_connected = Dense(62)(flatten)
    output = Dense(1, activation='sigmoid')(fully_connected)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=100, epochs=5 , validation_split=0.2)
    # TODO this works however the model sucks, find better architecture / loss functions etc.

    # Save the model
    model.save(cnn_model_path)


# KNN_model(pathname)
CNN_Model(pathname, 1)
