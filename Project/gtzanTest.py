import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from Project.model import Model

# DATA_PATH = "../../Datasets/GTZAN/images_original/"
from Project.model_custom import MyModel

# DATA_PATH = "../../Datasets/GTZAN/images_cropped/"
DATA_PATH = "../../Datasets/GTZAN/images_sliced/"

batch_size = 32
# img_height = 288
# img_width = 432
img_height = 217
# img_width = 16
img_width = 335
# img_height = 128
# img_width = 128
# img_height = 144
# img_width = 216

# noise_std = 0.2


def gaussian_noise_layer(input_layer, std):
    noise = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def slice_images_layer(x, y, slice_size):
    # slice = tf.shape(x)
    slice = x.numpy().shape
    print("image shape:", slice)
    return slice, y


def import_data():
    # tf.compat.v1.enable_eager_execution()
    data_dir = pathlib.Path(DATA_PATH)
    # image_count = len(list(data_dir.glob('*/*.jpg')))
    # print("Number of images:", image_count)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    # print(class_names)

    '''
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        first_image = image_batch[0]
        print(np.min(first_image), np.max(first_image))
        break
    '''

    print("Normalization...")

    # Gray-scale
    train_ds = train_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    val_ds = val_ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))

    # Slice train set
    train_ds = train_ds.map(lambda x, y: (slice_images_layer(x, y, 16)))


    # Scale to [0, 1]
    # normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    # train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Normalize images to 0 mean and 1 std
    train_ds = train_ds.map(lambda x, y: (tf.image.per_image_standardization(x), y))
    val_ds = val_ds.map(lambda x, y: (tf.image.per_image_standardization(x), y))

    # Add noise to 2 new dataset copies
    # train_ds_aug1 = train_ds.map(lambda x, y: (gaussian_noise_layer(x, noise_std), y))
    # train_ds_aug2 = train_ds.map(lambda x, y: (gaussian_noise_layer(x, noise_std * 2), y))

    # normalize them
    # train_ds_aug1 = train_ds_aug1.map(lambda x, y: (tf.image.per_image_standardization(x), y))
    # train_ds_aug2 = train_ds_aug2.map(lambda x, y: (tf.image.per_image_standardization(x), y))

    # concatenate to one big set
    # train_ds = train_ds.concatenate(train_ds_aug1).concatenate(train_ds_aug2)

    '''
    for images, labels in train_ds.take(1):
        # images_aug = images.numpy().tolist().map(lambda x: gaussian_noise_layer(x, 0.1))
        images_aug = gaussian_noise_layer(images, noise_std)
        images_aug = tf.convert_to_tensor(images_aug)
        images_aug = tf.image.per_image_standardization(images_aug)
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().reshape(img_height, img_width).astype("float32"), cmap='gray')
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images_aug[i].numpy().reshape(img_height, img_width).astype("float32"), cmap='gray')
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()
    '''

    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds_aug.take(1):

    '''
    for image_batch, labels_batch in train_ds:
        # print(image_batch.shape)
        # print(labels_batch.shape)
        first_image = image_batch[0]
        print(first_image.numpy().shape)
        print("train_ds:")
        print(np.min(first_image), np.max(first_image))
        print(np.mean(image_batch), np.std(image_batch))
        break

    for image_batch, labels_batch in val_ds:
        # print(image_batch.shape)
        # print(labels_batch.shape)
        first_image = image_batch[0]
        print("train_ds:")
        print(np.min(first_image), np.max(first_image))
        print(np.mean(image_batch), np.std(image_batch))
        break
    '''

    # AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache()  # .prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache()  # .prefetch(buffer_size=AUTOTUNE)
    # blues = list(data_dir.glob('blues/*'))
    # PIL.Image.open(str(blues[0]))

    return train_ds, val_ds


def useStuff():
    train_ds, val_ds = import_data()
    # ds_np = tfds.as_numpy(train_ds)
    # for dp in ds_np:
        # print(dp[0].shape)

    # print(train_ds[0].shape)
    # print(train_ds[1].shape)
    input_shape = (img_height, img_width, 1)
    '''
    model = Model(input_shape)
    model.trainTFDataset(train_ds, val_ds, 10)
    model.plotLearningCurve()
    '''
    # model = MyModel(input_shape)
    # model.trainFit(train_ds, val_ds, 10)
    # print(val_ds.shape)


if __name__ == '__main__':
    useStuff()
    # print(os.path.isdir(DATA_PATH))
