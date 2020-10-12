import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


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


# accuracy: 0.9887
def convNet(input_shape):
    l2Reg = 0.0001

    def conv_block(num_filters, filter_size, pooling=True):
        model.add(layers.Conv2D(num_filters, filter_size, kernel_regularizer=regularizers.l2(l2Reg)))
        # model.add(layers.Conv2D(num_filters, filter_size))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        if pooling:
            model.add(layers.MaxPooling2D((2, 2)))
            # model.add(layers.BatchNormalization())

    model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2Reg),
                            # input_shape=(256, 256, 1)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu',
                            input_shape=input_shape))
    # input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    conv_block(32, (3, 3))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2Reg)))
    # model.add(layers.MaxPooling2D((2, 2)))
    conv_block(32, (3, 3))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2Reg)))
    # model.add(layers.MaxPooling2D((2, 2)))
    conv_block(64, (3, 3))
    # model.add(layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(l2Reg), padding='same'))
    # model.add(layers.MaxPooling2D((2, 2), padding='same'))
    # conv_block(128, (5, 5))
    # model.add(layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(l2Reg), padding='same'))
    # model.add(layers.MaxPooling2D((2, 2), padding='same'))
    conv_block(64, (3, 3), pooling=True)
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(l2Reg), padding='same'))
    model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()
    return model
