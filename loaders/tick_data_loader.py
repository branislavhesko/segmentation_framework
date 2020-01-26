import cv2
import numpy as np

from loaders.data_loader_mask_generic import DataLoaderCrop2D


class TickDataLoaderCrop2D(DataLoaderCrop2D):
    MASK_CATEGORIES = {
        "COLON": (1, (0, 0.5, 0.5))
    }
        
    def __init__(self, img_files, mask_files=(), crop_size=(512, 512),
                 stride=0.1, transform=lambda x: x):
        super().__init__(img_files, mask_files, crop_size,
                 stride, transform)    
        
    def __getitem__(self, index):
        info = self.sub_image_info_holder.get_info_at_index(index)
        img = np.array(cv2.imread(info.img, cv2.IMREAD_COLOR)).astype(np.float32)
        img = img / 255.
        mask = cv2.imread(info.mask, cv2.IMREAD_COLOR) / 255.
        mask = self._transform_mask(mask)
        data = (*self._transform(*self._crop_image_and_mask(img, mask, info)),
                info.slice, info.img, info.mask)
        return data
    
    def _transform_mask(self, mask):
        new_mask = np.zeros(mask.shape[:2])
        for key, (index, color) in self.MASK_CATEGORIES.items():
            new_mask[(mask[:, :, 0] >= color[0]) & (mask[:, :, 1] >= color[1]) & (mask[:, :, 2] >= color[2])] = index
        return new_mask