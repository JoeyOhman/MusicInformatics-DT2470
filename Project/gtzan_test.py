import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from Project.image_utils import slice_images, normalize_images
from Project.load_gtzan import load_images
from Project.model import Model

# DATA_PATH = "../../Datasets/GTZAN/images_original/"
from Project.model_custom import MyModel

# DATA_PATH = "../../Datasets/GTZAN/images_cropped/"
# DATA_PATH = "../../Datasets/GTZAN/images_sliced/"

batch_size = 32
# img_height = 288
# img_width = 432
# img_height = 217
# img_height = 225
# img_width = 336
img_height = 256
img_width = 465
slice_width = 16
# img_height = 128
# img_width = 128
# img_height = 144
# img_width = 216

# noise_std = 0.2

'''
def gaussian_noise_layer(input_layer, std):
    noise = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise
'''


def import_data(val_split=0.2):
    print("Loading images...")
    data = load_images(img_height)
    print("Images loaded, shape:", data.shape)
    np.random.shuffle(data)

    # imgs = data[:, 0]
    # print(np.mean(imgs))

    # data[:, 0] = imgs

    n = len(data)
    num_val = int(n * val_split)
    train = data[: -num_val]
    val = data[-num_val:]

    '''
    x_t = train[:, 0]
    y_t = train[:, 1]
    print(x_t.shape)
    print(x_t[0].shape)

    x_t = np.dstack(x_t)
    print("dstack shape:", x_t.shape)
    # x_t = x_t.reshape((-1, img_height, img_width, 1))
    x_t = np.moveaxis(x_t, 2, 0)

    plt.imshow(x_t[0], cmap='gray')
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(x_t[i].reshape(img_height, img_width).astype("float32") * 255, cmap='gray')
        plt.title(y_t[i])
        plt.axis("off")
    plt.show()
    '''

    # Slice and Normalize
    print("Slicing and normalizing...")
    x_t, y_t, x_t_up, x_t_down = slice_images(train, img_height, slice_width)
    print(x_t.shape, y_t.shape, x_t_up.shape, x_t_down.shape)
    # x_t = np.vstack([x_t, x_t_up, x_t_down])
    # x_t += x_t_up + x_t_down
    x_t = np.concatenate([x_t, x_t_up, x_t_down])
    # y_t = np.vstack([y_t, y_t, y_t])  # Make sure this order is correct!
    y_t = np.concatenate([y_t, y_t, y_t])
    # y_t += y_t + y_t
    print(x_t.shape)
    normalize_images(x_t)
    # normalize_images(x_t_up)
    # normalize_images(x_t_down)

    x_t = np.dstack(x_t)
    x_t = np.moveaxis(x_t, 2, 0)
    x_t = np.expand_dims(x_t, 3)

    x_v, y_v, _, _ = slice_images(val, img_height, slice_width)
    normalize_images(x_v)

    x_v = np.dstack(x_v)
    x_v = np.moveaxis(x_v, 2, 0)
    x_v = np.expand_dims(x_v, 3)

    y_t = np.array(y_t, dtype="float32")
    y_v = np.array(y_v, dtype="float32")

    '''
    x_t = train_slices[:, 0]
    y_t = train_slices[:, 1]
    x_v = val[:, 0]
    y_v = val[:, 1]
    '''

    print(x_t.shape, y_t.shape, x_v.shape, y_v.shape)
    print(x_t.dtype, y_t.dtype, x_v.dtype, y_v.dtype)

    return x_t, y_t, x_v, y_v, val[:, 0], val[:, 1]


def import_and_train():
    # train_ds, val_ds = import_data()
    x_train, y_train, x_val, y_val, images_val_x, images_val_y = import_data()
    print(x_train[0].shape)
    print(y_train[0])
    # ds_np = tfds.as_numpy(train_ds)
    # for dp in ds_np:
    # print(dp[0].shape)
    # x_train = np.asarray(x_train).astype('float32')
    # y_train = np.asarray(y_train).astype('float32')
    # x_val = np.asarray(x_val).astype('float32')
    # y_val = np.asarray(y_val).astype('float32')

    # print(train_ds[0].shape)
    # print(train_ds[1].shape)
    input_shape = (img_height, slice_width, 1)
    model = Model(input_shape)
    model.evaluate_whole_images(images_val_x, images_val_y)
    model.train(x_train, y_train, x_val, y_val, 30, 32)
    model.evaluate_whole_images(images_val_x, images_val_y)
    model.plotLearningCurve()
    '''
    model = MyModel(input_shape)
    # model.trainFit(train_ds, val_ds, 10)
    model.trainFit(x_train, y_train, x_val, y_val, 3, 32)
    model.plotLearningCurve()
    '''
    # print(val_ds.shape)
    model.model.save("Models")


if __name__ == '__main__':
    # np.random.seed(123)
    import_and_train()
    # print(os.path.isdir(DATA_PATH))
