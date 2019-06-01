import pathlib
import sys; sys.path.append(pathlib.Path(__file__).parent.parent)
import glob
from tqdm import tqdm

import numpy as np
from PIL import Image


data_path = "./data/train/imgs"
imgs = glob.glob(data_path + "/*")

means = []
stds = []

for img in tqdm(imgs):
    i = Image.open(img)
    means.append(np.mean(np.asarray(i), axis=(0,1)))
    stds.append(np.std(np.asarray(i), axis=(0,1)))

print("mean: {}".format(np.mean(np.array(means), axis=0) / 255))
print("stds: {}".format(np.mean(np.array(stds), axis=0) / 255))
