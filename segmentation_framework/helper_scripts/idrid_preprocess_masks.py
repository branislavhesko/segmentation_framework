import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

from config import IdridSegmentation


PATH = "/home/brani/STORAGE/idrid/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/"
PATH_IMAGES = "/home/brani/STORAGE/idrid/A. Segmentation/1. Original Images/b. Testing Set/"
OUTPUT_PATH = "/home/brani/STORAGE/idrid/A. Segmentation/3. Processed Segmentations/b. Testing Set/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
folders = os.listdir(PATH)

masks = {
    folder: sorted(glob.glob(os.path.join(PATH, folder, "*.tif"))) for folder in folders
}


images = tqdm(glob.glob(os.path.join(PATH_IMAGES, "*.jpg")))
for image_file in images:
    images.set_description("Processing {}".format(image_file))
    basename = os.path.basename(image_file)[:-4]
    output = np.zeros_like(cv2.imread(image_file, cv2.IMREAD_COLOR))

    for mask_name, color in zip(masks.keys(), IdridSegmentation.COLORS):
        mask_file = list(filter(lambda x: basename in x, masks[mask_name]))
        if not len(mask_file):
            continue
        mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)
        output[mask > 0] = color
    cv2.imwrite(os.path.join(OUTPUT_PATH, "{}.png".format(basename)), output)
