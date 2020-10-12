from skimage import io, img_as_float
from PIL import Image
# import Image
import numpy as np
import os
import matplotlib.pyplot as plt

DATA_PATH = "../../Datasets/GTZAN/images_original/"
OUT_PATH = "../../Datasets/GTZAN/images_cropped/"


def cropImage(image_path):
    image = img_as_float(io.imread(image_path))

    # Select all pixels almost equal to white
    # (almost, because there are some edge effects in jpegs
    # so the boundaries may not be exactly white)
    white = np.array([1, 1, 1])
    mask = np.abs(image - white).sum(axis=2) < 0.50

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    out = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    out = out * 255
    out = out.astype(np.uint8)

    # plt.imshow(out)
    # plt.show()
    # imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = Image.fromarray(out)
    img.save(image_path)
    # img.show()


def convert_to_jpg(file_path):
    im = Image.open(file_path)
    rgb_im = im.convert('RGB')
    rgb_im.save(file_path.replace(".png", ".jpg").replace("images_original", "images_cropped"))
    # os.remove(file_path)


def traverse_images(data_path, image_operation):

    list_subfolders_with_paths = [(f.name, f.path) for f in os.scandir(data_path) if f.is_dir()]
    for sub_dir in list_subfolders_with_paths:
        dir_name, dir_path = sub_dir
        print(dir_name)
        for f in os.listdir(dir_path):
            file_path = dir_path + "/" + f
            if os.path.isfile(file_path):
                print(f)
                image_operation(file_path)


if __name__ == '__main__':
    traverse_images(DATA_PATH, convert_to_jpg)
    traverse_images(OUT_PATH, cropImage)
    # traverse_images(image_op)
