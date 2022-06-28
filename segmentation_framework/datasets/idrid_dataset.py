import cv2

from loaders.smart_random_data_loader import SmartRandomDataLoader


class IdridDataset(SmartRandomDataLoader):
    MASK_LOADER = cv2.IMREAD_COLOR

    def __init__(self, img_files, mask_files, crop_size, transform, config, **_):
        super(IdridDataset, self).__init__(
            config=config, img_files=img_files, mask_files=mask_files, crop_size=crop_size, transforms=transform)