from enum import Enum

from config import RefugeeDiscSegmentationConfig, RefugeeCupSegmentationConfig
from loaders.data_loader_mask_generic import DataLoaderCrop2D


class RefugeeSegmentationDataset(DataLoaderCrop2D):
    class Labels(Enum):
        BACKGROUND = 0
        OPTIC_DISC = 1
        OPTIC_CUP = 2

    def __init__(self, img_files, mask_files=(), crop_size=(512, 512),
                 stride=0.1, transform=lambda x: x):
        super().__init__(img_files, mask_files, crop_size, stride, transform, RefugeeCupSegmentationConfig())


if __name__ == "__main__":
    import glob
    img_files = sorted(glob.glob("../data/images/*.jpg"))
    mask_files = sorted(glob.glob("../data/GT_disc_cup/*.bmp"))

    dataset = RefugeeSegmentationDataset(img_files, mask_files, stride=0.5, transform=RefugeeDiscSegmentationConfig().AUGMENTATION)
    y = dataset[1]
    print(dataset[0])
