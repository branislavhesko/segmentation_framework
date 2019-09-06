import glob
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from config import Configuration
from loaders.data_loader_mask_generic import DataLoaderCrop2D


train_imgs = glob.glob("./data/train/imgs/*.png")
train_masks = glob.glob("./data/train/masks/*.png")

train_loader = DataLoaderCrop2D(train_imgs, train_masks, crop_size=(Configuration.CROP_SIZE, Configuration.CROP_SIZE),
                                stride=Configuration.STRIDE, transform=Configuration.AUGMENTATION)

val_imgs = glob.glob("./data/validate/imgs/*.png")
val_masks = glob.glob("./data/validate/masks/*.png")

val_loader = DataLoaderCrop2D(val_imgs, val_masks, 
                              crop_size=(Configuration.CROP_SIZE, Configuration.CROP_SIZE),
                              stride=Configuration.STRIDE, transform=Configuration.VAL_AUGMENTATION)
len_set = len(train_loader)
for i in range(100):
    idx = np.random.randint(0, len_set)
    data = train_loader[idx]
    img = data[0].permute([1, 2, 0]).cpu().numpy()
    mask = data[1].cpu().numpy()
    plt.figure(figsize=(16, 9), dpi=100)
    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(img))
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(mask))
    plt.savefig(str(i) + ".png", bbox_inches="tight")