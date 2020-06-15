from loaders.data_loader_mask_generic import DataLoaderCrop2D


class ThyroidDataset(DataLoaderCrop2D):

    def __init__(self, img_files, mask_files, crop_size, stride, transform, config):
        super().__init__(
            img_files=img_files, mask_files=mask_files, crop_size=crop_size,
            stride=stride, transform=transform, config=config)