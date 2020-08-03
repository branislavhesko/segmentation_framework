from collections import namedtuple
import cv2
import torch
from torch.utils.data import Dataset

from config import Configuration


CurrentlyOpened = namedtuple("CurrentlyOpened", ["image", "mask", "id"])


class SmartRandomDataLoader(Dataset):

    def __init__(self, config: Configuration, img_files, mask_files,
                 crop_size, transforms, num_random_crops_single_image, **_):
        self._img_files = img_files
        self._mask_files = mask_files
        self._crop_size = crop_size
        self._transforms = transforms
        self._num_random_crops = num_random_crops_single_image
        self._currently_opened = CurrentlyOpened(None, None, None)
        self._config = config

    def __len__(self):
        assert len(self._img_files) == len(self._mask_files)
        return len(self._img_files) * self._num_random_crops

    def __getitem__(self, item):
        image_id = item // self._num_random_crops
        if self._currently_opened.id != image_id:
            self._currently_opened = CurrentlyOpened(
                image=cv2.cvtColor(cv2.imread(self._img_files[image_id], cv2.IMREAD_COLOR) / 255., cv2.COLOR_BGR2RGB),
                mask=self._config.process_mask(cv2.imread(self._mask_files[image_id], cv2.IMREAD_GRAYSCALE)),
                id=image_id
            )
        rand_row, rand_col = self._get_random_crop(self._currently_opened.image.shape, self._crop_size)
        image_crop = self._currently_opened.image[rand_row: rand_row + self._crop_size[0],
                     rand_col: rand_col + self._crop_size[1], :]
        mask_crop = self._currently_opened.mask[rand_row: rand_row + self._crop_size[0],
                     rand_col: rand_col + self._crop_size[1]]
        data = (
            *self._transforms(image_crop, mask_crop),
            (rand_row, rand_col, rand_row + self._crop_size[0], rand_col + self._crop_size[1]),
            self._img_files[self._currently_opened.id], self._mask_files[self._currently_opened.id]
        )
        return data

    def _get_random_crop(self, image_size, crop_size):
        rand_row = torch.randint(low=0, high=image_size[0] - crop_size[0], size=[1])
        rand_col = torch.randint(low=0, high=image_size[1] - crop_size[1], size=[1])
        return rand_row.item(), rand_col.item()


if __name__ == "__main__":
    smart = SmartRandomDataLoader(Configuration(), None, None, None, None, None)
    for idx in range(100):
        print(smart._get_random_crop((1024, 1024), (256, 256)))
