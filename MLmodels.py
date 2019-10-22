"""
takes a number of ML models and saves them for the classification of letters
in to actual characters
"""


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
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, Add, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import keras


###################### Constants ###########################
pathname = '/cs/usr/punims/Desktop/OCRProject/Preprocessed_Images'
saved_models_path = '/cs/usr/punims/Desktop/OCRProject/MLmodels/knn_2n.gz'
path_to_filename = '/cs/usr/punims/Desktop/OCRProject/MLmodels/cnn1'
path_of_training_data = '/cs/usr/punims/Desktop/OCRProject/saved_data_sets/train_data.pkl'
path_of_training_labels ='/cs/usr/punims/Desktop/OCRProject/saved_data_sets/train_labels.pkl'
path_of_test_data = '/cs/usr/punims/Desktop/OCRProject/saved_data_sets/test_data.pkl'
path_of_test_labels = '/cs/usr/punims/Desktop/OCRProject/saved_data_sets/test_labels.pkl'

cnn_training_data_path = 'cnn_saved_data_sets/train_data.pkl'
cnn_test_data_path = 'cnn_saved_data_sets/test_data.pkl'
cnn_training_labels_path = 'cnn_saved_data_sets/train_labels.pkl'
cnn_test_labels_path = '/cs/usr/punims/Desktop/OCRProject/cnn_saved_data_sets/test_labels.pkl'
cnn_model_path = '/content/drive/My Drive/OCRProject/MLmodels/cnn_model.gz'

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

        X_train, X_test, y_train, y_test = train_test_split(features, labels_ascii)
        X_train = X_train.reshape(-1, 32, 32, 1)
        X_test = X_test.reshape(-1, 32, 32, 1)
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

        where_digits = np.where(np.logical_and(y>=48, y<=57))
        y[where_digits] = y[where_digits] - 48

        where_small_letters = np.where(np.logical_and(y>=97, y<=122))
        y[where_small_letters] = y[where_small_letters] - 61

        where_big_letters = np.where(np.logical_and(y>=65, y<=90))
        y[where_big_letters] = y[where_big_letters] - 55

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_train = X_train.reshape(-1, 32, 32, 1)
        X_test = X_test.reshape(-1, 32, 32, 1)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)



    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    #     model.add(Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
    #     model.add(Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
    model.add(Dense(62, activation='softmax'))
    print(y)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # Fit the model
    model.fit(X_test, y_test, validation_data=(X_test, y_test), epochs=100, batch_size=250, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    # This model seems to work well with a 3.3% base error (nearly 97% accuracy) 

    #     # CNN architecture OLD

    #     input = Input(shape=(32, 32, 1))
    #     conv1 = Conv2D(32, (3, 3))(input)
    #     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #     drop1 = Dropout(0.5)(pool1)
    #     conv2 = Conv2D(64, (3, 3))(drop1)
    #     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #     drop2 = Dropout(0.5)(pool2)
    # #     conv3 = Conv2D(64, (2, 2), activation='relu')(pool2)
    #     # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #     flatten = Flatten()(drop2)
    #     fully_connected = Dense(512, activation='sigmoid')(flatten)
    #     output = Dense(62, activation='sigmoid')(fully_connected)
    #     model = Model(inputs=input, outputs=output)
    #     model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
    #     print(model.summary())
    #     plt.imshow(X[5000].reshape(32,32), cmap='gray') # TODO delete
    #     plt.show()
    #     model.fit(X, y, epochs=10 , validation_split=0.3)
    #     # TODO this works however the model sucks, find better architecture / loss functions etc.

    # Save the model
    model.save(cnn_model_path)


# KNN_model(pathname)
CNN_Model(pathname, 0)

# if override_dataset: ANOTHER OLD MODEL
    #
    #     # Get images and labels
    #     feature_list, labels = extractFeatures.imagesAsIs(path)
    #     features = np.array(feature_list)
    #     labels_ascii = list(map(ord, labels))  #turns string labels in to ascii integer values.
    #     labels_ascii = np.array(labels_ascii)
    #
    # # split in to test and train datasets
    #
    #     # X_train, X_test, y_train, y_test = train_test_split(features, labels_ascii)
    #     X = features
    #     y = labels_ascii
    #     X = X.reshape(-1, 32, 32, 1)
    #     # X_test = X_test.reshape(-1, 32, 32, 1)
    #
    #     # create save files for the data
    #     joblib.dump(X, cnn_training_data_path)
    #     joblib.dump(y, cnn_training_labels_path)
    #     # joblib.dump(X_test, cnn_test_data_path)
    #     # joblib.dump(y_test, cnn_test_labels_path)
    #
    # else:
    #     X = joblib.load(cnn_training_data_path)
    #     y = joblib.load(cnn_training_labels_path)
    #
    # # CNN architecture
    # input = Input(shape=(32, 32, 1))
    # conv1 = Conv2D(6, (5, 5), activation='relu')(input)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(16, (5, 5), activation='relu')(pool1)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv3 = Conv2D(64, (2, 2), activation='relu')(pool2)
    # # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # flatten = Flatten()(conv3)
    # fully_connected = Dense(120, activation='relu')(flatten)
    # fully_connected = Dense(84, activation='relu')(flatten)
    # output = Dense(62, activation='softmax')(fully_connected)
    # model = Model(inputs=input, outputs=output)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # # plt.imshow(X[5000].reshape(32,32), cmap='gray') # TODO delete
    # # plt.show()
    # model.fit(X, y, batch_size=1000, epochs=5 , validation_split=0.2)
    # # TODO this works however the model sucks, find better architecture / loss functions etc.
    #
    # # Save the model
    # model.save(cnn_model_path)


# KNN_model(pathname)
CNN_Model(pathname, 0)
