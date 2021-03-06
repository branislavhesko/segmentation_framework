from enum import Enum

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from losses.api import FocalTverskyLoss
from models.combine_net import CombineNet
from models.deeplab import DeepLab
from models.psp_net import PSPNet
from models.refinenet.refinenet_4cascade import RefineNet4Cascade
from models.unet import UNet
from models.u2net import U2NET
from utils.transforms import (ComposeTransforms, Normalize, RandomHorizontalFlip, RandomRotate,
                              RandomVerticalFlip, RandomSquaredCrop, ToTensor, Transpose)


class NetMode(Enum):
    TRAIN = "TRAIN"
    VALIDATE = "VALIDATE"


class ImagesSubfolder(Enum):
    IMAGES = "imgs/*.tif"
    MASKS = "masks/*.tif"


class DataProps:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]


available_models = {
    "CombineNet": CombineNet,
    "DeepLabV3p": DeepLab,
    "PSPNet": PSPNet,
    "RefineNet": RefineNet4Cascade,
    "UNet": UNet,
    "U2Net": U2NET
}


class Configuration:
    FOCAL_LOSS_INDICES = None
    CE_LOSS_INDICES = None
    BATCH_SIZE = 2
    CHECKPOINT = ""
    SAVE_FREQUENCY = 4
    CLASS_VALUE = -1
    CROP_SIZE = 256
    CUDA = True
    DATASET = {
        NetMode.TRAIN: "SmartRandomDataLoader",
        NetMode.VALIDATE: "DataLoaderCrop2D",
    }
    FOLDER_WITH_IMAGE_DATA = "/home/branislav/datasets/refuge"

    LEARNING_RATE = 1e-4
    LOSS = CrossEntropyLoss

    MODEL = "DeepLabV3p"
    NUM_CLASSES = 2
    NUM_WORKERS = 8
    NUMBER_OF_EPOCHS = 100
    OUTPUT = "ckpt"
    OUTPUT_FOLDER = "polyps"
    STRIDE = 0.5
    STRIDE_VAL = 0.5
    STRIDE_LIMIT = (1000, 0.5)  # THIS PREVENTS DATASET HALTING
    
    OPTIMALIZER = SGD
    VALIDATION_FREQUENCY = 1  # num epochs
    
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    AUGMENTATION = ComposeTransforms([
        RandomRotate(0.6),
        RandomSquaredCrop(0.85),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        Transpose(),
        ToTensor()
    ])
    VAL_AUGMENTATION = ComposeTransforms([
        Transpose(),
        ToTensor()
    ])
    PATH_TO_SAVED_SUBIMAGE_INFO = None  # FOLDER_WITH_IMAGE_DATA + "train/info.pkl"

    FOLDERS = {
        NetMode.TRAIN: "train",
        NetMode.VALIDATE: "train"
    }
    SUBFOLDERS = {
        ImagesSubfolder.IMAGES: "images/*.tif",
        ImagesSubfolder.MASKS: "mask/*.tif"
    }
    NUM_RANDOM_CROPS_PER_IMAGE = 4
    VISUALIZER = "VisualizationSaveImages"

    def serialize(self):
        output = {}
        for key in list(filter(lambda x: x.isupper(), dir(self))):
            value = getattr(self, key)
            if any(map(lambda type_: isinstance(value, type_), [str, float, int, tuple, list, dict])):
                output[key] = str(value)
        return output

    def __str__(self):
        serialized = self.serialize()
        return "\n".join([f"{key}: {value}" for key, value in serialized.items()])

    def process_mask(self, mask):
        mask[mask > 0] = 1
        return mask


class IdridSegmentation(Configuration):
    LOSS = FocalTverskyLoss
    LOSS_PARAMS = {
    }
    CHECKPOINT = "U2Net_epoch5__12-22-2020_03_53_08_NUM_CLASSES6_mean_loss0.598_accuracy0.983_mean_IOU0.631_mean_DICE0.729.pth"
    NUM_CLASSES = 6
    FOLDER_WITH_IMAGE_DATA = "/home/branislav/other/idrid/A. Segmentation/"
    FOLDERS = {
        NetMode.TRAIN: "train",
        NetMode.VALIDATE: "eval"
    }
    SUBFOLDERS = {
        ImagesSubfolder.IMAGES: "images/*jpg",
        ImagesSubfolder.MASKS: "masks/*png"
    }
    MODEL = "U2Net"
    CROP_SIZE = 512
    COLORS = (
        [255, 0, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
        [255, 255, 0]
    )
    DATASET = {
        NetMode.TRAIN: "IdridDataset",
        NetMode.VALIDATE: "DataLoaderCrop2D",
    }
    PATH_TO_SAVED_SUBIMAGE_INFO = "/home/branislav/other/idrid/A. Segmentation/eval.pickle"
    BATCH_SIZE = 2
    NUM_WORKERS = 4
    NUM_RANDOM_CROPS_PER_IMAGE = 200
    VALIDATION_FREQUENCY = 5
    STRIDE = 1.
    STRIDE_VAL = 1.
    STRIDE_LIMIT = (1000, 1.)
    OUTPUT_FOLDER = "IDRID_U2Net"
    LEARNING_RATE = 1e-5
    FOCAL_LOSS_INDICES = (1, 2, 4)
    CE_LOSS_INDICES = (0, 3, 5)
    NUMBER_OF_EPOCHS = 31

    def process_mask(self, mask):
        output_mask = np.zeros(mask.shape[:2])
        for idx, color in enumerate(self.COLORS):
            output_mask[np.all(mask == color, axis=-1)] = idx + 1
        return output_mask


class TickColonSegmentation(Configuration):
    CHECKPOINT = ""
    NUM_CLASSES = 2
    OUTPUT_FOLDER = "tick"
    DATASET = "TickDataLoaderCrop2D"
    VISUALIZER = "VisualizationTensorboard"


class RefugeeDiscSegmentationConfig(Configuration):
    CHECKPOINT = ""
    NUM_CLASSES = 2
    CLASS_VALUE = 128
    OUTPUT_FOLDER = "refugee_disc"
    DATASET = "DataLoaderCrop2D"
    FOLDERS = {
        NetMode.TRAIN: "train",
        NetMode.VALIDATE: "validate"
    }
    SUBFOLDERS = {
        ImagesSubfolder.IMAGES: "images/*.jpg",
        ImagesSubfolder.MASKS: "masks/*.bmp"
    }
    STRIDE_VAL = 0.5
    STRIDE = 0.5
    STRIDE_LIMIT = (2000, 0.5)
    PATH_TO_SAVED_SUBIMAGE_INFO = "/home/branislav/datasets/refuge/refugee_data.pickle"
    CUDA = True
    VISUALIZER = "VisualizationTensorboard"

    def process_mask(self, mask):
        mask[mask <= self.CLASS_VALUE] = 1
        mask[mask > self.CLASS_VALUE] = 0
        return mask


class RefugeeCupSegmentationConfig(RefugeeDiscSegmentationConfig):
    CLASS_VALUE = 10
    OUTPUT_FOLDER = "refugee_cup"
    PATH_TO_SAVED_SUBIMAGE_INFO = None
    CHECKPOINT = "CombineNet_epoch10__05-18-2020_06_20_30_NUM_CLASSES2_mean_loss0.008_accuracy0.997_mean_IOU0.881_mean_DICE0.912.pth"


class ThyroidConfig(Configuration):
    CROP_SIZE = 256
    BATCH_SIZE = 4
    CHECKPOINT = ""
    NUM_CLASSES = 2
    CLASS_VALUE = 128
    OUTPUT_FOLDER = "thyroid"
    DATASET = "ThyroidDataset"
    FOLDERS = {
        NetMode.TRAIN: "train",
        NetMode.VALIDATE: "validate"
    }
    SUBFOLDERS = {
        ImagesSubfolder.IMAGES: "images/*.PNG",
        ImagesSubfolder.MASKS: "masks/*.PNG"
    }
    FOLDER_WITH_IMAGE_DATA = "/home/brani/STORAGE/DATA/uzv_thyroid/"
    STRIDE = 1.
    STRIDE_VAL = 1.
    STRIDE_LIMIT = (2000, 1.)
    CUDA = True
    VISUALIZER = "VisualizationTensorboard"
    PATH_TO_SAVED_SUBIMAGE_INFO = "/home/brani/STORAGE/DATA/uzv_thyroid/thyroid.pickle"


    def process_mask(self, mask):
        return (mask > 0).astype(np.int32)


if __name__ == "__main__":
    cfg = RefugeeDiscSegmentationConfig()
    print(cfg.serialize())
