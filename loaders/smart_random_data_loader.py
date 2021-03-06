from collections import namedtuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Configuration

CurrentlyOpened = namedtuple("CurrentlyOpened", ["image", "mask", "id"])


class SmartRandomDataLoader(Dataset):
    MASK_LOAD_TYPE = cv2.IMREAD_COLOR

    def __init__(self, config: Configuration, img_files, mask_files,
                 crop_size, transforms, **_):
        self._config = config
        self._img_files = img_files
        self._imgs = [cv2.cvtColor(cv2.imread(img_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255. for img_file in self._img_files]
        self._mask_files = mask_files
        self._masks = [self._config.process_mask(cv2.imread(mask, self.MASK_LOAD_TYPE)) for mask in self._mask_files]
        self._crop_size = crop_size
        self._transforms = transforms
        self._num_random_crops = self._config.NUM_RANDOM_CROPS_PER_IMAGE
        self._currently_opened = CurrentlyOpened(None, None, None)

    def __len__(self):
        assert len(self._img_files) == len(self._mask_files)
        return len(self._img_files) * self._num_random_crops

    def __getitem__(self, item):
        image_id = item // self._num_random_crops
        if self._currently_opened.id != image_id:
            self.assign_currently_opened(image_id)
        rand_row, rand_col = self._get_random_crop(self._currently_opened.image.shape, self._crop_size)
        image_crop, mask_crop = self._crop(rand_col, rand_row)
        data = (
            *self._transforms(image_crop, mask_crop),
            (rand_row, rand_col, rand_row + self._crop_size[0], rand_col + self._crop_size[1]),
            self._img_files[self._currently_opened.id], self._mask_files[self._currently_opened.id]
        )
        return data

    def _crop(self, rand_col, rand_row):
        image_crop = np.copy(self._currently_opened.image[rand_row: rand_row + self._crop_size[0],
                                                  rand_col: rand_col + self._crop_size[1], :])
        mask_crop = np.copy(self._currently_opened.mask[rand_row: rand_row + self._crop_size[0],
                                                rand_col: rand_col + self._crop_size[1]])
        return image_crop, mask_crop

    def assign_currently_opened(self, image_id):
        self._currently_opened = CurrentlyOpened(
            image=self._imgs[image_id],
            mask=self._masks[image_id],
            id=image_id
        )

    def _get_random_crop(self, image_size, crop_size):
        rand_row = torch.randint(low=0, high=image_size[0] - crop_size[0], size=[1])
        rand_col = torch.randint(low=0, high=image_size[1] - crop_size[1], size=[1])
        return rand_row.item(), rand_col.item()


if __name__ == "__main__":
    smart = SmartRandomDataLoader(Configuration(), None, None, None, None, None)
    for idx in range(100):
        print(smart._get_random_crop((1024, 1024), (256, 256)))
