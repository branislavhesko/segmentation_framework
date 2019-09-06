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
    NUM_CLASSES = 2
    BATCH_SIZE = 2
    CROP_SIZE = 256
    STRIDE = 0.5
    STRIDE_VAL = 0.2
    STRIDE_LIMIT = (1000, 1.)  # THIS PREVENTS DATASET HALTING
    NUMBER_OF_EPOCHS = 100
    LEARNING_RATE = 1e-3
    FOLDER_WITH_IMAGE_DATA = "./data/"
    OUTPUT = "ckpt"
    OUTPUT_FOLDER = "vessels_segmentation_unnormalized"
    
    MODEL = "CombineNet"
    CHECKPOINT = ""
    LOSS = CrossEntropyLoss
    OPTIMALIZER = SGD
    VALIDATION_FREQUENCY = 5  # num epochs
    CUDA = True
    
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    AUGMENTATION = ComposeTransforms([
        #Normalize(DataProps.MEAN, DataProps.STD),
        RandomSquaredCrop(0.9),
        RandomRotate(0.5, std_dev=10),
        RandomHorizontalFlip(),
        Transpose(),
        ToTensor()
    ])
    VAL_AUGMENTATION = ComposeTransforms([
        #Normalize(DataProps.MEAN, DataProps.STD),
        Transpose(),
        ToTensor()
    ])
    PATH_TO_SAVED_SUBIMAGE_INFO = None  # FOLDER_WITH_IMAGE_DATA + "train/info.pkl"
