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
    CHECKPOINT = "CombineNet_epoch36__09-21-2019_23_09_55_NUM_CLASSES2_mean_loss0.121_accuracy0.975_mean_IOU0.792_mean_DICE0.839.pth"

    CROP_SIZE = 256
    CUDA = True
    DATASET = "DataLoaderCrop2D"
    FOLDER_WITH_IMAGE_DATA = "./data"

    LEARNING_RATE = 1e-4
    LOSS = CrossEntropyLoss

<<<<<<< HEAD
    MODEL = "CombineNet"
=======
    MODEL = "DeepLabV3p"
>>>>>>> b190a59210f116028110c0bf9a353372b09d1ebd
    NUM_CLASSES = 2
    NUM_WORKERS = 2
    NUMBER_OF_EPOCHS = 100
    OUTPUT = "ckpt"
    OUTPUT_FOLDER = "polyps"
    STRIDE = 1
<<<<<<< HEAD
    STRIDE_VAL = 0.5
    STRIDE_LIMIT = (1000, 1)  # THIS PREVENTS DATASET HALTING
    
    OPTIMALIZER = SGD
    VALIDATION_FREQUENCY = 5  # num epochs
=======
    STRIDE_VAL = 1
    STRIDE_LIMIT = (1000, 1)  # THIS PREVENTS DATASET HALTING
    
    OPTIMALIZER = SGD
    VALIDATION_FREQUENCY = 2  # num epochs
>>>>>>> b190a59210f116028110c0bf9a353372b09d1ebd
    
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


class TickColonSegmentation(Configuration):
    CHECKPOINT = ""
    NUM_CLASSES = 2
    OUTPUT_FOLDER = "tick"
    DATASET = "TickDataLoaderCrop2D"
    VISUALIZER = "VisualizationTensorboard"
