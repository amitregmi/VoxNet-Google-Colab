"""
Trains and evaluates a 3D CNN on ModelNet10.
See below for usage.
"""

import sys
import numpy as np
np.random.seed(1)

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers.convolutional import Conv3D

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle

# Load the data
modelnet_file = "C:\Users\Amit_Regmi\Desktop\Amit2015\Artificial Intelligence\Python Scripts\data\modelnet10.npz"
data = np.load(modelnet_file)
X, Y = shuffle(data['X_train'], data['y_train'])
X_test, Y_test = shuffle(data['X_test'], data['y_test'])

# One-hot encode training targets
Y = keras.utils.to_categorical(Y, num_classes=10)

# Build the network
model = Sequential()
model.add(Reshape((30, 30, 30, 1), input_shape=(30, 30, 30)))  # 1 in-channel
model.add(Conv3D(16, 6, strides=2, activation='relu'))
model.add(Conv3D(64, 5, strides=2, activation='relu'))
model.add(Conv3D(64, 5, strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#Loading Weights
filename = "C:\Users\Amit_Regmi\Desktop\Amit2015\Artificial Intelligence\Python Scripts\modelnet-cnn-master\data\\VoxNet_Weight.hdf5"
model.load_weights(filename)

# Train
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4))
model.fit(X, Y, batch_size=256, epochs=30, verbose=2,validation_split=0.2, shuffle=True)

# Show test accuracy
Y_test_pred = np.argmax(model.predict(X_test), axis=1)
print('Test accuracy: {:.3f}'.format(accuracy_score(Y_test, Y_test_pred)))

# Show confusion matrix and average per-class accuracy
conf = confusion_matrix(Y_test, Y_test_pred)
avg_per_class_acc = np.mean(np.diagonal(conf) / np.sum(conf, axis=1))
print('Confusion matrix:\n{}'.format(conf))
print('Average per-class accuracy: {:.3f}'.format(avg_per_class_acc))
