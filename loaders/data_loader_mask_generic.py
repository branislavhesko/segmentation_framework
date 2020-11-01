import logging
import pickle

import numpy as np
import cv2
import os

from config import Configuration, NetMode
from loaders.subimage_info_holder import InfoEnum, SubImageInfoHolder


class DataLoaderCrop2D:
    MASK_LOAD_TYPE = cv2.IMREAD_COLOR

    def __init__(self, img_files, mask_files=(), crop_size=(512, 512),
                 stride=0.1, transform=lambda x: x, config: Configuration = Configuration(), mode=NetMode.VALIDATE):
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

        path = self._config.PATH_TO_SAVED_SUBIMAGE_INFO.replace(".pickle", f"_{mode.value}.pickle")
        if path is not None and os.path.exists(
                path) and self._is_subimage_file_valid(path):
            self.sub_image_info_holder.load_info_dict(path)
        else:
            self.sub_image_info_holder.fill_info_dict()
            if self._config.PATH_TO_SAVED_SUBIMAGE_INFO is not None:
                self.sub_image_info_holder.save_info_dict(path)

    def _is_subimage_file_valid(self, path):
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        logger = logging.getLogger(self.__class__.__name__)
        if data["config"] == self._config.serialize():
            logger.info("Config is equal to serialized dataset info, allowing loading it!")
            return True
        logger.info("Config is not equal to serialized dataset info, creating new config!")
        return False

    def __len__(self):
        return len(self.sub_image_info_holder.info_dict[InfoEnum.INDEX])

    def __getitem__(self, index):
        info = self.sub_image_info_holder.get_info_at_index(index)
        img = cv2.cvtColor(cv2.imread(info.img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img / 255.
        mask = cv2.imread(info.mask, self.MASK_LOAD_TYPE)
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
