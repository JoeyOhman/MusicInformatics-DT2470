import numpy as np


def slice_image_no_label(img, img_height, slice_width):
    sample = [img, -1, img, img]
    return np.array(slice_image(sample, img_height, slice_width))[:, 0]


def slice_image(sample, img_height, slice_width):
    x, y, x_aug_up, x_aug_down = sample
    img_w = x.shape[1]
    # assert img_w % slice_width == 0  # Assume images are padded to even number of slices

    slices = []
    for s_idx in range(0, img_w, slice_width):
        s = x[:, s_idx: s_idx + slice_width]
        s_up = x_aug_up[:, s_idx: s_idx + slice_width]
        s_down = x_aug_down[:, s_idx: s_idx + slice_width]
        if s.shape[1] < slice_width or s_up.shape[1] < slice_width or s_down.shape[1] < slice_width:
            break

        assert s.shape == (img_height, slice_width)
        assert s_up.shape == (img_height, slice_width)
        assert s_down.shape == (img_height, slice_width)
        slices.append([s, y, s_up, s_down])

    return slices


def slice_images(data, img_height, slice_width):
    slices = []
    n = len(data)

    for i in range(n):
        slices += slice_image(data[i], img_height, slice_width)

    slices = np.array(slices)
    return slices[:, 0], slices[:, 1], slices[:, 2], slices[:, 3]


def normalize_image(img):
    adjusted_stddev = max(np.std(img), 1.0 / np.sqrt(img.shape[0] * img.shape[1]))
    img = (img - np.mean(img)) / adjusted_stddev
    return img


def normalize_images(imgs):
    for i in range(len(imgs)):
        imgs[i] = normalize_image(imgs[i])
    return imgs
