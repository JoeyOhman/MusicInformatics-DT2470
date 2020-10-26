import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix

from Project.image_utils import slice_images, normalize_images
from Project.load_gtzan import load_images
from Project.model import Model

batch_size = 32
img_height = 256
img_width = 465
slice_width = 16


def import_data(val_split=0.2):
    print("Loading images...")
    train, val = load_images(img_height)

    # Slice and Normalize
    print("Slicing and normalizing...")
    x_t, y_t, x_t_up, x_t_down = slice_images(train, img_height, slice_width)
    print(x_t.shape, y_t.shape, x_t_up.shape, x_t_down.shape)

    x_t = np.concatenate([x_t, x_t_up, x_t_down])
    y_t = np.concatenate([y_t, y_t, y_t])
    print(x_t.shape)
    normalize_images(x_t)

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
    x_train, y_train, x_val, y_val, images_val_x, images_val_y = import_data()
    print(x_train[0].shape)
    print(y_train[0])

    input_shape = (img_height, slice_width, 1)
    model = Model(input_shape)
    model.evaluate_whole_images(images_val_x, images_val_y)
    model.train(x_train, y_train, x_val, y_val, 5, 32)
    model.evaluate_whole_images(images_val_x, images_val_y)
    model.plotLearningCurve()
    '''
    model = MyModel(input_shape)
    # model.trainFit(train_ds, val_ds, 10)
    model.trainFit(x_train, y_train, x_val, y_val, 3, 32)
    model.plotLearningCurve()
    '''
    # print(val_ds.shape)
    model.model.save("ModelsFiltered")


def visualize_filters(model):
    first_conv_layer = model.model.layers[0]

    filters, biases = first_conv_layer.get_weights()
    print(first_conv_layer.name, filters.shape)

    # Normalize filter values to 0-1
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n = 4
    f, axarr = plt.subplots(n, n)
    filter_counter = 0
    for x in range(n):
        for y in range(n):
            axarr[x, y].set_xticks([])
            axarr[x, y].set_yticks([])
            axarr[x, y].imshow(np.squeeze(filters[:, :, :, filter_counter]), cmap="gray")
            filter_counter += 1

    plt.show()


def plot_confusion_mat(model):
    x_train, y_train, x_val, y_val, images_val_x, images_val_y = import_data()
    images_val_y = np.array([int(y) for y in images_val_y])
    predictions = np.array(model.predict_whole_images(images_val_x))
    predictions = np.array([int(y) for y in predictions])
    print("Predictions:")
    print(predictions)
    print("Labels:")
    print(images_val_y)

    print("Shapes:")
    print(images_val_y.shape, predictions.shape)
    conf_mat = confusion_matrix(images_val_y, predictions)

    classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    df_cm = pd.DataFrame(conf_mat, classes, classes)
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

    plt.show()


if __name__ == '__main__':
    # import_and_train()
    model = Model(input_shape=(img_height, slice_width, 1), model_path="ModelsFiltered")
    # visualize_filters(model)
    plot_confusion_mat(model)
