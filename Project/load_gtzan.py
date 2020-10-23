import numpy as np
import os
from skimage import io, img_as_float
import matplotlib.pyplot as plt

# DATA_PATH = "../../Datasets/GTZAN/images_cropped/"
DATA_PATH = "../../Datasets/GTZAN/mel_spectrograms/"

classes = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]


def pad_image(img, img_height):
    height, width = img.shape
    # num_cols_to_pad = 16 - width % 16
    # img = np.hstack((img, np.zeros((height, num_cols_to_pad))))
    num_rows_to_pad = img_height % height
    img = np.vstack((np.zeros((num_rows_to_pad, img.shape[1])), img))
    return img


def load_image(img_height, dir_path, f):
    file_path = dir_path + "/" + f
    if os.path.isfile(file_path):
        image = img_as_float(io.imread(file_path, as_gray=True))
        image = np.asarray(image).astype('float32')
        return pad_image(image, img_height)
    else:
        return None


def load_images(img_height):
    genre_to_class = dict(zip(classes, range(len(classes))))
    data = []
    data_path = DATA_PATH

    list_subfolders_with_paths = [(f.name, f.path) for f in os.scandir(data_path) if f.is_dir()]
    for sub_dir in list_subfolders_with_paths:
        dir_name, dir_path = sub_dir
        if "_aug" in dir_name:
            continue
        label = genre_to_class[dir_name]
        data_dir = []
        for f in os.listdir(dir_path):
            image = load_image(img_height, dir_path, f)
            if image is not None:
                data_dir.append([image, label])

        # Loop through corresponding augmented directory
        aug_up_dir_path = dir_path.replace(dir_name, dir_name + "_augup")
        aug_down_dir_path = dir_path.replace(dir_name, dir_name + "_augdown")
        for idx, f in enumerate(os.listdir(aug_up_dir_path)):
            image = load_image(img_height, aug_up_dir_path, f)
            if image is not None:
                data_dir[idx].append(image)
        for idx, f in enumerate(os.listdir(aug_down_dir_path)):
            image = load_image(img_height, aug_down_dir_path, f)
            if image is not None:
                data_dir[idx].append(image)

        data += data_dir

    return np.array(data)


if __name__ == '__main__':
    dataset = load_images()
    print(dataset.shape)
