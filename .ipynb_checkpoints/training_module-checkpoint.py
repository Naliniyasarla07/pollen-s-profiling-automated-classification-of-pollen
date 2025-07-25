# training_module.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10  # replace with your dataset

def train_cnn(epochs=5, batch_size=32):
    """
    Trains a simple CNN model and returns:
     - 'model': the trained Keras model object
     - 'loss': list of training loss per epoch
     - 'accuracy': list of training accuracy per epoch
    """
    # Replace this with your actual data-loading logic
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)

    input_shape = x_train.shape[1:]  # e.g. (32,32,3)
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    return {
        'model': model,
        'loss': history.history['loss'],
        'accuracy': history.history['accuracy']
    }
