import tensorflow as tf
from tensorflow.keras import layers, models


def feedForwardNet():
    model = tf.keras.models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])
    model.summary()

    return model


def convNet(input_shape):

    def conv_block(num_filters, filter_size, pooling=True):
        model.add(layers.Conv2D(num_filters, filter_size, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        if pooling:
            model.add(layers.MaxPooling2D((2, 2), padding="same"))
            model.add(layers.BatchNormalization())

    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), input_shape=input_shape, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.BatchNormalization())

    conv_block(64, (5, 5))
    conv_block(64, (5, 5), pooling=True)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10))

    model.summary()
    return model
