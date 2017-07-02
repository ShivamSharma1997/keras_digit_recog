# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Importing Dependenceis And DataSet
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

#Loading The Data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Fixing Ramdom Seed
np.random.seed(1)

#flatten 28*28 images to as vactor for each image
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

#Normalize Inputs to 0-1
x_train = x_train/255
x_test = x_test/255

#one hot encode output
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
#print(y_train)


def baseline_model():
    #Create Model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    #Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#Building The Model
model = baseline_model()

#Fitting The Model
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, batch_size = 200, verbose = 1)

#Final Evaluation
scores = model.evaluate(x_test, y_test, verbose = 0)
print("Baseline Error: %.2f" % (100-scores[1]*100))