import numpy as np
import cv2
from skimage.transform import resize, rotate
import torch


class RandomHorizontalFlip:

    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, img, mask):
        if self._p < np.random.rand():
            return img[:, ::-1, :], mask[:, ::-1]
        else:
            return img, mask


class RandomVerticalFlip:

    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, img, mask):
        if self._p < np.random.rand():
            return img[::-1, :, :], mask[::-1, :]
        else:
            return img, mask


class RandomSquaredCrop:

    def __init__(self, minimal_relative_crop_size=0.5):
        self._minimal_relative_crop_size = minimal_relative_crop_size

    def __call__(self, img, mask):
        h, w = mask.shape
        size = np.random.randint(int(self._minimal_relative_crop_size * h), h)

        crop_x_origin = np.random.randint(0, h - size - 1)
        crop_y_origin = np.random.randint(0, w - size - 1)
        print("Crop_x: {}, crop_y: {}, size: {}".format(crop_x_origin, crop_y_origin, size))
        return (
            img[crop_x_origin: crop_x_origin + size, crop_y_origin: crop_y_origin + size, :],
            mask[crop_x_origin: crop_x_origin + size, crop_y_origin: crop_y_origin + size])


class RandomRotate:
    def __init__(self, p=0.5, std_dev=10):
        self._p = p
        self._std_dev = std_dev

    def __call__(self, img, mask):
        if np.random.rand() > self._p:
            return img, mask
        else:
            # angle = np.random.randint(0, 180)
            angle = np.random.normal(0, self._std_dev)  # normal distribution
            return rotate(img, angle), rotate(mask, angle)
            

class Resize:
    def __init__(self, size):
        self._size = size

    def __call__(self, img, mask):
        return resize(img, self._size), resize(mask, self._size)


class ComposeTransforms:
    def __init__(self, transforms):
        self._transforms = transforms
    
    def __call__(self, img, mask):
        for transform in self._transforms:
            img, mask = transform(img, mask)
        
        return img, mask


class Normalize:
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std
        

class ToTensor:

    def __call__(self, img, mask):
        return torch.from_numpy(img), torch.from_numpy(mask)
        


if __name__ == "__main__":
    img = cv2.imread("./Study_02_00007_01_L_registered.avi_average_image.tif")

    mask = cv2.cvtColor(cv2.threshold(img, 100, 255, type=cv2.THRESH_BINARY)[1], cv2.COLOR_BGR2GRAY)

    transform = RandomRotate(1)
    from matplotlib import pyplot
    for i in range(20):
        pyplot.imshow(transform(img, mask)[0])
        pyplot.show(block=False)