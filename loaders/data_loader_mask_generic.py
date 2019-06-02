import numpy as np
import cv2
import os
from PIL import Image

from loaders.subimage_info_holder import get_number_of_subimages, ImageLoader, InfoEnum, SubImageInfoHolder


# TODO: it remains to make better border image indexing
class DataLoaderCrop2D:

    def __init__(self, img_files, mask_files=(), crop_size=(512, 512),
                 stride=0.1, transform=lambda x: x):
        self._img_files = img_files
        self._mask_files = mask_files
        self._stride = stride
        self._crops_one_image = 0
        self._index_of_opened_file = 0
        self._transform = transform
        self._crop_size = crop_size
        self.sub_image_info_holder = SubImageInfoHolder(self._img_files, self._mask_files, self._crop_size, self._stride)
        self.sub_image_info_holder.fill_info_dict()

    def __len__(self):
        return len(self.sub_image_info_holder.info_dict[InfoEnum.INDEX])

    def __getitem__(self, index):
        info = self.sub_image_info_holder.get_info_at_index(index)
        img = cv2.imread(info.img, cv2.IMREAD_COLOR)
        img = img.transpose([2, 0, 1]) / 255.
        mask = cv2.imread(info.mask, cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1
        data = (*self._transform(*self._crop_image_and_mask(img, mask, info)), info.slice)
        return data

    @staticmethod
    def _crop_image_and_mask(img, mask, info):
        slice = info.slice
        if len(img.shape) > 2: 
            return (img[:, slice[0] : slice[2], slice[1]: slice[3]],
                mask[slice[0] : slice[2], slice[1]: slice[3]])
        else:
            return (img[:, slice[0] : slice[2], slice[1]: slice[3]],
                mask[slice[0] : slice[2], slice[1]: slice[3]])



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
