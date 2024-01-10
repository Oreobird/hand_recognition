

import os

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout

import pickle

work_dir = '/content/drive/MyDrive/hand_recognition/'
data_dir = os.path.join(work_dir, 'data/')

def load_data():
    with open(os.path.join(data_dir, 'X_test.pkl'), 'rb') as file:
        X_test = pickle.load(file)

    with open(os.path.join(data_dir, 'y_test.pkl'), 'rb') as file:
        y_test = pickle.load(file)

    with open(os.path.join(data_dir, 'X_train.pkl'), 'rb') as file:
        X_train = pickle.load(file)

    with open(os.path.join(data_dir, 'y_train.pkl'), 'rb') as file:
        y_train = pickle.load(file)

    return X_train, y_train, X_test, y_test

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    model.summary()
    return model

if __name__ == "__main__":
    model_file = os.path.join(work_dir, 'trained_model/hand_rec_model.h5')

    X_train, y_train, X_test, y_test = load_data()
    model = create_model()
    history = model.fit(X_train, y_train, epochs=1, batch_size=96, verbose=1, validation_data=(X_test, y_test))
    model.save(model_file)

    model = tf.keras.models.load_model(model_file)
    tf.saved_model.save(model, os.path.join(work_dir, "trained_model/tmp_model"))
