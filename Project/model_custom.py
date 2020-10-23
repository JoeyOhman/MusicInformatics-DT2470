import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import Model
# from tensorflow.keras.utils import Progbar

from Project.networks import convNet, feedForwardNet

metrics_names = ['acc', 'loss']


class MyModel(Model):

    def get_config(self):
        pass

    def __init__(self, input_shape):
        super(MyModel, self).__init__()
        self.model = init_model(input_shape)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

        self.history = None

        # self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # self.optimizer = tf.keras.optimizers.Adam()

        # self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        # self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def call(self, x, **kwargs):
        x = self.model(x)
        return x

    @tf.function
    def train_step(self, data):
        print("inside train_step!")
        images, labels = data

        # SPLIT DATA INTO SLICES
        # PREDICT EACH SLICE AND TRAIN FOR EACH

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self(images, training=True)
            # loss = self.loss_object(labels, predictions)
            loss = self.compiled_loss(labels, predictions, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # self.train_loss(loss)
        # self.train_accuracy(labels, predictions)

        self.compiled_metrics.update_state(labels, predictions)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        print("inside TEST_step!")
        images, labels = data

        # SPLIT DATA INTO SLICES
        # PREDICT EACH SLICE AND USE "MAJORITY VOTE"

        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self(images, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(labels, predictions, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(labels, predictions)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    # def trainFit(self, train_ds, val_ds, epochs):
    def trainFit(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        self.history = self.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size)

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
    '''
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    probModel = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probModel.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    '''

    return model  # , probModel
