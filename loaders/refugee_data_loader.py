import cv2
import numpy as np
import torch

from config import RefugeeCupDiscSegmentationConfig
from loaders.data_loader_mask_generic import DataLoaderCrop2D


class RefugeeSegmentationDataset(DataLoaderCrop2D):

    def __init__(self, img_files, mask_files=(), crop_size=(512, 512),
                 stride=0.1, transform=lambda x: x):
        super().__init__(img_files, mask_files, crop_size, stride, transform, RefugeeCupDiscSegmentationConfig())


if __name__ == "__main__":
    import os
    import glob
    img_files = sorted(glob.glob("../data/images/*.jpg"))
    mask_files = sorted(glob.glob("../data/GT_disc_cup/*.bmp"))

    dataset = RefugeeSegmentationDataset(img_files, mask_files, stride=0.5, transform=RefugeeCupDiscSegmentationConfig().AUGMENTATION)
    print(dataset[0])