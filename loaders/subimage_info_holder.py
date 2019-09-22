from collections import namedtuple
from enum import auto, Enum
import os
import pickle
import sys
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

from config import Configuration


class InfoEnum(Enum):
    FILE_NAME_IMG = auto()
    FILE_NAME_MASK = auto()
    INDEX = auto()
    NUMBER_OF_INDICES = auto()
    SLICE_INDICES = auto()


class SubImageInfoHolder:

    Info = namedtuple("Info", ["img", "mask", "img_number", "slice", "index"])

    def __init__(self, img_files, mask_files, crop_size, stride):
        self._img_files = img_files
        self._stride = stride
        self._mask_files = mask_files
        assert len(self._img_files) == len(self._mask_files)
        self.image_loader = ImageLoader(crop_size, stride)
        self.info_dict = {
            InfoEnum.FILE_NAME_IMG: [],
            InfoEnum.FILE_NAME_MASK: [],
            InfoEnum.INDEX: [],
            InfoEnum.NUMBER_OF_INDICES: [],
            InfoEnum.SLICE_INDICES: []
        }

    def fill_info_dict(self):
        for image_number, (img, mask) in tqdm(enumerate(list(zip(self._img_files, self._mask_files)))):
            img_matrix = np.asarray(cv2.imread(img, cv2.IMREAD_COLOR))
            mask_matrix = np.asarray(cv2.imread(mask, cv2.IMREAD_GRAYSCALE))
            if img_matrix.shape[0] > Configuration.STRIDE_LIMIT[0]:
                stride = Configuration.STRIDE_LIMIT[1]
            else:
                stride = Configuration.STRIDE
            self.image_loader.set_image_and_mask(img_matrix, mask_matrix, stride)
            indices = self.image_loader.get_indices()
            for index in indices:
                self.info_dict[InfoEnum.FILE_NAME_IMG].append(img)
                self.info_dict[InfoEnum.FILE_NAME_MASK].append(mask)
                self.info_dict[InfoEnum.INDEX].append(image_number)
                self.info_dict[InfoEnum.NUMBER_OF_INDICES].append(len(indices))
                self.info_dict[InfoEnum.SLICE_INDICES].append(index)
        if Configuration.PATH_TO_SAVED_SUBIMAGE_INFO is not None:
            self.save_info_dict(Configuration.PATH_TO_SAVED_SUBIMAGE_INFO)

    def save_info_dict(self, filename):
        file = open(filename, "wb")
        pickle.dump(self.info_dict, file)
        file.close()

    def load_info_dict(self, filename):
        file = open(filename, "rb")
        self.info_dict = pickle.load(file)

    def __getitem__(self, index):
        return self.get_info_at_index(index)

    def __len__(self):
        return len(self.info_dict[InfoEnum.FILE_NAME_IMG])

    def get_info_at_index(self, index):
        info = self.Info(
            img=self.info_dict[InfoEnum.FILE_NAME_IMG][index],
            mask=self.info_dict[InfoEnum.FILE_NAME_MASK][index],
            img_number=self.info_dict[InfoEnum.INDEX][index],
            slice=self.info_dict[InfoEnum.SLICE_INDICES][index],
            index=index
            )
        return info

    def __str__(self):
        return "Actual number of sub-images in the set: {}, dictionary size is: {} MB".format(
            len(self.info_dict[InfoEnum.INDEX]), sys.getsizeof(self.info_dict[InfoEnum.FILE_NAME_IMG]) * 5. / 1000000.)


class ImageLoader:

    def __init__(self, crop_size, stride):
        self._image = None
        self._mask = None
        self._crop_size = crop_size
        self._stride = stride
        self._indices = []
        # self.get_indices()

    def __len__(self):
        return np.multiply(*get_number_of_subimages(self._image.shape, self._crop_size, self._stride))

    def __getitem__(self, index):
        indices = self._indices[index]
        #print(indices)
        return (
            self._image[indices[0]:indices[0] + indices[2], indices[1]:indices[1] + indices[3], :],
            self._mask[indices[0]:indices[0] + indices[2], indices[1]:indices[1] + indices[3]],
            indices
        )

    def get_indices(self):
        num_subimages = get_number_of_subimages(self._image.shape, self._crop_size, self._stride)
        row_add = int(self._crop_size[0] * self._stride)
        col_add = int(self._crop_size[1] * self._stride)
        self._indices = []
        for i in range(num_subimages[0]):
            for j in range(num_subimages[1]):
                if i * row_add + self._crop_size[0] > self._image.shape[0]:
                    row = self._image.shape[0] - self._crop_size[0]
                else:
                    row = int(i * row_add)
                if j * col_add + self._crop_size[1] > self._image.shape[1]:
                    col = self._image.shape[1] - self._crop_size[1]
                else:
                    col = int(j * col_add)
                indices = (int(row), int(col), int(row) + self._crop_size[0], (int(col) + self._crop_size[1]), 1)
                row_reversed = self._image.shape[0] - int(row)
                col_reversed = self._image.shape[1] - int(col)
                indices_reversed = (row_reversed - self._crop_size[0], col_reversed - self._crop_size[1], 
                                    row_reversed, col_reversed, -1)
                self._indices.append(indices)
                self._indices.append(indices_reversed)
        return self._indices

    def set_image_and_mask(self, image, mask, stride=None):
        if stride is not None:
            self._stride = stride
        self._image = image
        self._mask = mask
        assert self._image.shape[:2] == self._mask.shape


def get_number_of_subimages(shape, crop_size, stride):
    y_dir = int((shape[0] - crop_size[0]) // (crop_size[0] * stride) + 2)
    x_dir = int((shape[1] - crop_size[1]) // (crop_size[1] * stride) + 2)
    #print(y_dir)
    #print(y_dir)
    return np.array((y_dir, x_dir))


if __name__ == "__main__":
    import glob
    imgs = glob.glob("./train/imgs/*.png")
    masks = glob.glob("./train/masks/*.png")
    imgs.sort()
    masks.sort()

    sub = SubImageInfoHolder(imgs, masks, (512, 512), 0.2)

    sub.fill_info_dict()

    for i in range(1000):
        print(sub.get_info_at_index(i))
    print(sub)