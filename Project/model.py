import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers


class Model:

    def __init__(self, ):
        self.model, self.probModel = init_model()
        self.history = None

    def train(self, x, y, epochs, x_test=None, y_test=None):
        if x_test is None:
            self.history = self.model.fit(x, y, epochs=epochs)
        else:
            self.history = self.model.fit(x, y, epochs=epochs, validation_data=(x_test, y_test))

    def evaluate(self, x, y):
        self.model.evaluate(x, y, verbose=2)

    def predict(self, x):
        return self.probModel(x).numpy()

    def plotLearningCurve(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        if 'val_accuracy' in self.history.history:
            plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.8, 1])
        plt.legend(loc='lower right')
        plt.show()


# 20 epochs:
# train_acc: 0.9819
# test_acc: 0.9786
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
def convNet():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                            input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001), padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(128, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.0001), padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10))

    model.summary()
    return model


def init_model():
    # model = feedForwardNet()
    model = convNet()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    probModel = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probModel.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    return model, probModel
