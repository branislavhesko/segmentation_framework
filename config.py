from enum import Enum

import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from models.combine_net import CombineNet
from models.deeplab import DeepLab
from models.psp_net import PSPNet
from models.unet import UNet
from utils.learning_rate import adaptive_learning_rate
from utils.transforms import (Clahe, ComposeTransforms, Normalize, RandomHorizontalFlip, RandomRotate, 
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
    "UNet": UNet
}


class Configuration:
    BATCH_SIZE = 2
    CHECKPOINT = ""
    SAVE_FREQUENCY = 10
    CLASS_VALUE = -1
    CROP_SIZE = 512
    CUDA = True
    DATASET = "DataLoaderCrop2D"
    FOLDER_WITH_IMAGE_DATA = "./data"

    LEARNING_RATE = 1e-4
    LOSS = CrossEntropyLoss

    MODEL = "CombineNet"
    NUM_CLASSES = 2
    NUM_WORKERS = 2
    NUMBER_OF_EPOCHS = 100
    OUTPUT = "ckpt"
    OUTPUT_FOLDER = "polyps"
    STRIDE = 0.5
    STRIDE_VAL = 0.5
    STRIDE_LIMIT = (1000, 0.5)  # THIS PREVENTS DATASET HALTING
    
    OPTIMALIZER = SGD
    VALIDATION_FREQUENCY = 2  # num epochs
    
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    AUGMENTATION = ComposeTransforms([
        Normalize(DataProps.MEAN, DataProps.STD),
        RandomRotate(0.6),
        RandomSquaredCrop(0.85),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        Transpose(),
        ToTensor()
    ])
    VAL_AUGMENTATION = ComposeTransforms([
        Normalize(DataProps.MEAN, DataProps.STD),
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
    VISUALIZER = "VisualizationSaveImages"

    def serialize(self):
        output = {}
        for key in list(filter(lambda x: x.isupper(), dir(self))):
            value = getattr(self, key)
            if any(map(lambda type_: isinstance(value, type_), [str, float, int, tuple, list, dict])):
                output[key] = str(value)
        return output

    def process_mask(self, mask):
        mask[mask > 0] = 1
        return mask


class TickColonSegmentation(Configuration):
    CHECKPOINT = "CombineNet_epoch84__03-14-2020_09_43_01_NUM_CLASSES2_mean_loss0.068_accuracy0.975_mean_IOU0.888_mean_DICE0.926.pth"
    NUM_CLASSES = 2
    OUTPUT_FOLDER = "tick"
    DATASET = "TickDataLoaderCrop2D"
    VISUALIZER = "VisualizationTensorboard"


class RefugeeDiscSegmentationConfig(Configuration):
    CHECKPOINT = ""
    NUM_CLASSES = 2
    CLASS_VALUE = 128
    OUTPUT_FOLDER = "refugee_cup"
    DATASET = "RefugeeDataset"
    FOLDERS = {
        NetMode.TRAIN: "train",
        NetMode.VALIDATE: "validate"
    }
    SUBFOLDERS = {
        ImagesSubfolder.IMAGES: "images/*.jpg",
        ImagesSubfolder.MASKS: "masks/*.bmp"
    }
    FOLDER_WITH_IMAGE_DATA = "../data"
    STRIDE_VAL = 0.5
    STRIDE = 0.5
    STRIDE_LIMIT = (2000, 1.)
    PATH_TO_SAVED_SUBIMAGE_INFO = "../data/refugee_data.pickle"
    CUDA = True
    VISUALIZER = "VisualizationTensorboard"

    def process_mask(self, mask):
        mask[mask <= self.CLASS_VALUE] = 1
        mask[mask > self.CLASS_VALUE] = 0
        return mask


class RefugeeCupSegmentationConfig(RefugeeDiscSegmentationConfig):
    CLASS_VALUE = 10
    OUTPUT_FOLDER = "refugee_cup_correct"
    PATH_TO_SAVED_SUBIMAGE_INFO = None


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
    STRIDE = 0.5
    STRIDE_VAL = 0.5
    STRIDE_LIMIT = (2000, 1.)
    CUDA = True
    VISUALIZER = "VisualizationTensorboard"

    def process_mask(self, mask):
        return (mask > 0).astype(np.int32)


if __name__ == "__main__":
    cfg = RefugeeDiscSegmentationConfig()
    print(cfg.serialize())
