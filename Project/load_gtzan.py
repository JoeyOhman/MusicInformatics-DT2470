import numpy as np
import os
from skimage import io, img_as_float
import matplotlib.pyplot as plt

# DATA_PATH = "../../Datasets/GTZAN/images_cropped/"
DATA_PATH = "../../Datasets/GTZAN/mel_spectrograms/"
PARTITION_PATH = "../../Partitioning/"

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

partition_dict_is_train = {}


def load_partition_dict():
    file_train = PARTITION_PATH + "train_filtered.txt"
    file_test = PARTITION_PATH + "test_filtered.txt"

    with open(file_train) as f:
        lines = f.readlines()
        print("Num train lines:", len(lines))
        for line in lines:
            sample_name = line.strip().split("/")[1].replace(".wav", "")
            partition_dict_is_train[sample_name] = True

    with open(file_test) as f:
        lines = f.readlines()
        print("Num test lines:", len(lines))
        for line in lines:
            sample_name = line.strip().split("/")[1].replace(".wav", "")
            partition_dict_is_train[sample_name] = False


def is_image_train(file_name):
    sample_name = file_name.split(".")
    sample_name = sample_name[0] + "." + sample_name[1][:5]
    # print(sample_name)
    return partition_dict_is_train.get(sample_name, None)


def pad_image(img, img_height):
    height, width = img.shape
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
    load_partition_dict()
    genre_to_class = dict(zip(classes, range(len(classes))))
    data_t = []
    data_v = []
    data_path = DATA_PATH

    list_subfolders_with_paths = [(f.name, f.path) for f in os.scandir(data_path) if f.is_dir()]
    for sub_dir in list_subfolders_with_paths:
        dir_name, dir_path = sub_dir
        if "_aug" in dir_name:
            continue
        label = genre_to_class[dir_name]
        data_dir_t = []
        data_dir_v = []
        for f in os.listdir(dir_path):
            image = load_image(img_height, dir_path, f)
            if image is not None:
                sample = [image, label]
                is_train = is_image_train(f)
                if is_train is None:
                    continue
                if is_train:
                    data_dir_t.append(sample)
                else:
                    data_dir_v.append(sample)

        # Loop through corresponding augmented directory
        aug_up_dir_path = dir_path.replace(dir_name, dir_name + "_augup")
        aug_down_dir_path = dir_path.replace(dir_name, dir_name + "_augdown")

        count_t = 0
        count_v = 0
        for idx, f in enumerate(os.listdir(aug_up_dir_path)):
            image = load_image(img_height, aug_up_dir_path, f)
            if image is not None:
                is_train = is_image_train(f)
                if is_train is None:
                    continue
                if is_train:
                    data_dir_t[count_t].append(image)
                    count_t += 1
                else:
                    data_dir_v[count_v].append(image)
                    count_v += 1

        count_t = 0
        count_v = 0
        for idx, f in enumerate(os.listdir(aug_down_dir_path)):
            image = load_image(img_height, aug_down_dir_path, f)
            if image is not None:
                is_train = is_image_train(f)
                if is_train is None:
                    continue
                if is_train:
                    data_dir_t[count_t].append(image)
                    count_t += 1
                else:
                    data_dir_v[count_v].append(image)
                    count_v += 1
                # data_dir[idx].append(image)

        data_t += data_dir_t
        data_v += data_dir_v

    return np.array(data_t), np.array(data_v)


if __name__ == '__main__':
    train, test = load_images(256)
