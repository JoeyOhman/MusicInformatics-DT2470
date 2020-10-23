import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
import numpy as np

from Project.image_utils import slice_image, normalize_images, slice_image_no_label
from Project.networks import convNet, feedForwardNet


class Model:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model, self.prob_model = init_model(input_shape)
        self.history = None

    def train(self, x, y, x_test, y_test, epochs, batch_size=32):
        self.history = self.model.fit(x, y, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)

    def trainTFDataset(self, train_ds, val_ds, epochs):
        self.history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    def evaluate(self, x, y):
        self.model.evaluate(x, y, verbose=2)

    def predict(self, x):
        # predict here instead to make sure dropouts are turned off?
        return self.prob_model(x).numpy()

    def plotLearningCurve(self):
        acc_hist = self.history.history['accuracy']
        plt.plot(range(1, len(acc_hist) + 1), acc_hist, label='accuracy')
        if 'val_accuracy' in self.history.history:
            val_acc_hist = self.history.history['val_accuracy']
            plt.plot(range(1, len(val_acc_hist) + 1), val_acc_hist, label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

    def predict_whole_images(self, x):
        img_height, slice_width, _ = self.input_shape
        predictions = []
        n = len(x)
        for i in range(n):
            slices = slice_image_no_label(x[i], x[i].shape[0], slice_width)
            slices = normalize_images(slices)
            slices = np.dstack(slices)
            # x_v = x_v.reshape((-1, img_height, slice_width, 1))
            slices = np.moveaxis(slices, 2, 0)
            slices = np.expand_dims(slices, 3)
            probs = self.prob_model.predict(slices)
            sum_probs = np.sum(probs, axis=0)
            final_prediction = np.argmax(sum_probs)
            predictions.append(final_prediction)
        return predictions

    def evaluate_whole_images(self, x, y):
        predictions = self.predict_whole_images(x)
        # print(predictions)
        # print(y)
        num_correct = np.sum(predictions == y)
        print("Accuracy sum-rule:", "{:.2f}".format((np.round(100 * num_correct / len(x), decimals=2))) + "%")


def init_model(input_shape):
    # model = feedForwardNet()
    model = convNet(input_shape)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    prob_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    prob_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    return model, prob_model
