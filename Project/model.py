import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers

from Project.networks import convNet, feedForwardNet


class Model:

    def __init__(self, input_shape):
        self.model, self.probModel = init_model(input_shape)
        self.history = None

    def train(self, x, y, epochs, x_test=None, y_test=None):
        if x_test is None:
            self.history = self.model.fit(x, y, epochs=epochs, batch_size=32)
        else:
            self.history = self.model.fit(x, y, epochs=epochs, validation_data=(x_test, y_test))

    def trainTFDataset(self, train_ds, val_ds, epochs):
        self.history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    def evaluate(self, x, y):
        self.model.evaluate(x, y, verbose=2)

    def predict(self, x):
        # predict here instead to make sure dropouts are turned off?
        return self.probModel(x).numpy()

    def plotLearningCurve(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        if 'val_accuracy' in self.history.history:
            plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # plt.ylim([0.8, 1])
        plt.legend(loc='lower right')
        plt.show()


def init_model(input_shape):
    # model = feedForwardNet()
    model = convNet(input_shape)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    probModel = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probModel.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    return model, probModel
