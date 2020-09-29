import os
from time import time

import tensorflow as tf
import matplotlib.pyplot as plt

# from tensorflow.python.client import device_lib
from Project.model import Model

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def loadData():
    mnist = tf.keras.datasets.mnist

    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Scale images to pixel values [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape, add third dimension for each point, to fit CNN which can handle images with R.G.B for example
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    return x_train, y_train, x_test, y_test


def doStuff():
    x_train, y_train, x_test, y_test = loadData()
    # plt.imshow(x_test[0], cmap='gray')
    # plt.show()
    model = Model()
    startTime = time()
    model.train(x_train, y_train, 20, x_test=x_test, y_test=y_test)
    endTime = time()
    print("Training time:", endTime - startTime)
    model.evaluate(x_test, y_test)
    model.plotLearningCurve()


if __name__ == '__main__':
    doStuff()
    # print(device_lib.list_local_devices())
