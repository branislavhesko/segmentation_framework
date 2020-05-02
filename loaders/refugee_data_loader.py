from enum import Enum

from config import RefugeeCupDiscSegmentationConfig
from loaders.data_loader_mask_generic import DataLoaderCrop2D


class RefugeeSegmentationDataset(DataLoaderCrop2D):
    class Labels(Enum):
        BACKGROUND = 0
        OPTIC_DISC = 1
        OPTIC_CUP = 2

    def __init__(self, img_files, mask_files=(), crop_size=(512, 512),
                 stride=0.1, transform=lambda x: x):
        super().__init__(img_files, mask_files, crop_size, stride, transform, RefugeeCupDiscSegmentationConfig())

    def _process_mask(self, mask):
        mask[mask == 0] = 1
        mask[mask == 128] = 2
        mask[mask == 255] = 0
        return mask


if __name__ == "__main__":
    import glob
    img_files = sorted(glob.glob("../data/images/*.jpg"))
    mask_files = sorted(glob.glob("../data/GT_disc_cup/*.bmp"))

    dataset = RefugeeSegmentationDataset(img_files, mask_files, stride=0.5, transform=RefugeeCupDiscSegmentationConfig().AUGMENTATION)
    y = dataset[1]
    print(dataset[0])
