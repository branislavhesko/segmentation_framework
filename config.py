import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from models.combine_net import CombineNet
from utils.learning_rate import adaptive_learning_rate
from utils.transforms import (ComposeTransforms, Normalize, RandomHorizontalFlip, RandomRotate, 
                              RandomVerticalFlip, RandomSquaredCrop, ToTensor)


class DataProps:
    MEAN = [0.54479471, 0.2145689,  0.07742358]
    STD = [0.29776531, 0.13578865, 0.04884564]


available_models = {
    "CombineNet": CombineNet,
    "DeepLabV3p": None
}


class Configuration:
    NUM_CLASSES = 2
    BATCH_SIZE = 1
    CROP_SIZE = 512
    STRIDE = 0.2
    STRIDE_LIMIT = (1000, 0.4)  # THIS PREVENTS DATASET HALTING
    NUMBER_OF_EPOCHS = 1
    LEARNING_RATE = 1e-2 / np.sqrt(16 / 2)
    FOLDER_WITH_IMAGE_DATA = "./data/"
    OUTPUT = "ckpt"
    OUTPUT_FOLDER = "vessels_segmentation"
    
    MODEL = "CombineNet"
    CHECKPOINT = ""
    LOSS = CrossEntropyLoss
    OPTIMALIZER = SGD
    VALIDATION_FREQUENCY = 2  # num epochs
    CUDA = True
    
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    AUGMENTATION = ComposeTransforms([
        Normalize(DataProps.MEAN, DataProps.STD),
        RandomRotate(1., std_dev=10),
        RandomHorizontalFlip(),
        ToTensor()
    ])
    VAL_AUGMENTATION = ComposeTransforms([
        Normalize(DataProps.MEAN, DataProps.STD),
        ToTensor()
    ])
    PATH_TO_SAVED_SUBIMAGE_INFO = None  # FOLDER_WITH_IMAGE_DATA + "train/info.pkl"
