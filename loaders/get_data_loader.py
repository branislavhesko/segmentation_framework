import glob
import os
from torch.utils.data import DataLoader

from config import Configuration, ImagesSubfolder, NetMode
from loaders.available_datasets import AvailableDatasets


def get_data_loaders(config: Configuration):
    imgs_train, masks_train = _get_imgs_masks_files(config, NetMode.TRAIN)
    imgs_val, masks_val = _get_imgs_masks_files(config, NetMode.VALIDATE)

    dataloader_train = AvailableDatasets.DATASETS[config.DATASET](img_files=imgs_train, mask_files=masks_train,
                                                                  crop_size=(config.CROP_SIZE, config.CROP_SIZE),
                                                                  stride=config.STRIDE, transform=config.AUGMENTATION,
                                                                  config=config, mode=NetMode.TRAIN)
    dataloader_val = AvailableDatasets.DATASETS[config.DATASET](img_files=imgs_val, mask_files=masks_val,
                                                                crop_size=(config.CROP_SIZE, config.CROP_SIZE),
                                                                stride=config.STRIDE_VAL,
                                                                transform=config.VAL_AUGMENTATION,
                                                                config=config, mode=NetMode.VALIDATE)
    loader_train = DataLoader(dataloader_train, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)
    # TODO: currently only validation with batch_size 1 is supported
    loader_val = DataLoader(dataloader_val, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    return loader_train, loader_val


def _get_imgs_masks_files(config, netmode=NetMode.VALIDATE):
    imgs_val = sorted(glob.glob(os.path.join(
        config.FOLDER_WITH_IMAGE_DATA, config.FOLDERS[netmode],
        config.SUBFOLDERS[ImagesSubfolder.IMAGES])))
    masks_val = sorted(glob.glob(os.path.join(
        config.FOLDER_WITH_IMAGE_DATA, config.FOLDERS[netmode],
        config.SUBFOLDERS[ImagesSubfolder.MASKS])))
    return imgs_val, masks_val
