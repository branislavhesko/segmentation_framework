import pickle

import numpy as np
import cv2
import os

from config import Configuration
from loaders.subimage_info_holder import InfoEnum, SubImageInfoHolder


class DataLoaderCrop2D:

    def __init__(self, img_files, mask_files=(), crop_size=(512, 512),
                 stride=0.1, transform=lambda x: x, config: Configuration = Configuration()):
        self._img_files = img_files
        self._mask_files = mask_files
        self._stride = stride
        self._crops_one_image = 0
        self._index_of_opened_file = 0
        self._transform = transform
        self._crop_size = crop_size
        self._config = config
        self.sub_image_info_holder = SubImageInfoHolder(
            self._img_files, self._mask_files, self._crop_size, self._stride, config=self._config)
        if self._config.PATH_TO_SAVED_SUBIMAGE_INFO is not None and os.path.exists(
                self._config.PATH_TO_SAVED_SUBIMAGE_INFO) and self._is_subimage_file_valid():
            self.sub_image_info_holder.load_info_dict(self._config.PATH_TO_SAVED_SUBIMAGE_INFO)
        else:
            self.sub_image_info_holder.fill_info_dict()
        print(self.sub_image_info_holder)

    def _is_subimage_file_valid(self):
        with open(self._config.PATH_TO_SAVED_SUBIMAGE_INFO, "rb") as fp:
            data = pickle.load(fp)
        if data["config"] == self._config.serialize():
            print("Config is equal to serialized dataset info, allowing loading it!")
            return True
        print("Config is not equal to serialized dataset info, creating new config!")
        return False

    def __len__(self):
        return len(self.sub_image_info_holder.info_dict[InfoEnum.INDEX])

    def __getitem__(self, index):
        info = self.sub_image_info_holder.get_info_at_index(index)
        img = np.array(cv2.imread(info.img, cv2.IMREAD_COLOR)).astype(np.float32)
        img = img / 255.
        mask = cv2.imread(info.mask, cv2.IMREAD_GRAYSCALE)
        mask = self._config.process_mask(mask)
        data = (*self._transform(*self._crop_image_and_mask(img, mask, info)),
                info.slice, info.img, info.mask)
        return data

    @staticmethod
    def _crop_image_and_mask(img, mask, info):
        slice_ = info.slice
        if len(img.shape) > 2: 
            return (img[slice_[0]: slice_[2], slice_[1]: slice_[3], :],
                    mask[slice_[0]: slice_[2], slice_[1]: slice_[3]])
        else:
            return (img[slice_[0]: slice_[2], slice_[1]: slice_[3]],
                    mask[slice_[0]: slice_[2], slice_[1]: slice_[3]])


def reconstruct_image(subimages, indices, original_shape, count_map=None):
    if not count_map:
        count_map = np.zeros(original_shape)
        for index in indices:
            count_map[index[0]: index[2], index[1]: index[3]] += 1

    image = np.zeros(original_shape)

    for subimage, index in zip(subimages, indices):
        image[index[0]: index[0] + index[2], index[1]: index[1] + index[3]] += subimage

    return np.divide(image, count_map)


if __name__ == "__main__":
    import glob
    imgs = glob.glob("./train/imgs/*.png")
    masks = glob.glob("./train/masks/*.png")
    loader = DataLoaderCrop2D(imgs, masks, stride=0.2)
    from matplotlib import pyplot as plt
    for i in range(100):
        plt.imshow(loader[i * 100][0])
        plt.show()
