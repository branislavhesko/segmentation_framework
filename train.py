from collections import namedtuple
from datetime import datetime
import glob
import os
import shutil
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat
import torch
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from config import available_models, Configuration
from loaders.data_loader_mask_generic import DataLoaderCrop2D
from models.combine_net import CombineNet
from stats.stats_meter import StatsMeter
from utils.segmentation_vizualization import (
    generate_palette, map_palette, show_segmentation_into_original_image, vizualize_segmentation)

class TrainModel:
    
    def __init__(self, config: Configuration):
        self._config = config
        self.model = None
        self.loss = None
        self.optimizer = None
        self.loader_val = None
        self.loader_train = None
        self.average_meter_train = StatsMeter(self._config.NUM_CLASSES)
        self.average_meter_val = StatsMeter(self._config.NUM_CLASSES)

        self._initialize()
        self._load_weights_if_available()

    def train(self):
        self.model.train()
        for epoch in range(self._config.NUMBER_OF_EPOCHS):
            for data in tqdm(self.loader_train):
                img, mask, indices, img_path, mask_path = data
                self.optimizer.zero_grad()
                img = Variable(img)
                mask = Variable(mask)
                if self._config.CUDA:
                    img = img.cuda()
                    mask = mask.cuda()

                output = self.model(img)
                prediction = torch.argmax(output, dim=1)
                loss = self.loss(output, mask)
                loss.backward()
                self.optimizer.step()
                self.average_meter_train.update(prediction.cpu().numpy(), mask.cpu().numpy(), loss.item())

            print("\n" + "-" * 50 + "\n")
            print(self.average_meter_train)
            print("\n" + "-" * 50 + "\n")
            if epoch % self._config.VALIDATION_FREQUENCY == 0:
                self.validate(epoch)
                self.model.train()
                self.save_model()

    def validate(self, epoch_num=0):
        path_to_save = os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER, str(epoch_num) + "_epoch")
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
            
        self.model.eval()
        self.average_meter_val = StatsMeter(self._config.NUM_CLASSES)
        img_shape = None
        CurrentlyOpened = namedtuple("CurrentlyOpened", ["img", "mask"], verbose=False)
        opened = CurrentlyOpened(None, None)
        count_map = None
        output_segmented = None

        for data in tqdm(self.loader_val):
            img, mask, indices, img_path, mask_path = data

            if self._config.CUDA:
                img = img.cuda()
                mask = mask.cuda()

            if opened.img != img_path[0]:
                if count_map is not None and output_segmented is not None:
                    self._save_segmentation(
                        output_segmented.cpu().numpy(), count_map.cpu().numpy(), 
                        opened.img, opened.mask, path_to_save)
                opened = CurrentlyOpened(img_path[0], mask_path[0])
                img_shape = cv2.imread(opened.img, cv2.IMREAD_GRAYSCALE).shape
                output_segmented = torch.zeros((self._config.NUM_CLASSES, img_shape[0], img_shape[1])).cuda()
                count_map = torch.zeros(img_shape).cuda()
            
            output = self.model(img)
            prediction = torch.argmax(output, dim=1)
            loss = self.loss(output, mask)

            # TODO: print val stats...
            self.average_meter_val.update(prediction.cpu().numpy(), mask.cpu().numpy(), loss.item())

            output_segmented[:, indices[0]: indices[2], indices[1]: indices[3]] += output[0, :, :, :].data
            count_map[indices[0]: indices[2], indices[1]: indices[3]] += 1
        self._save_segmentation(
            output_segmented.cpu().numpy(), count_map.cpu().numpy(),
            opened.img, opened.mask, path_to_save)
        print("\n" + "-" * 50 + "\n")
        print(self.average_meter_val)
        print("\n" + "-" * 50 + "\n")

    def _initialize(self):
        self.model = available_models[self._config.MODEL](self._config.NUM_CLASSES)
        if self._config.CUDA:
            self.model.cuda()
        print(self.model)
        self.loss = self._config.LOSS(size_average=True)
        self.optimizer = self._config.OPTIMALIZER([
            {'params': [param for name, param in self.model.named_parameters() if name[-4:] == 'bias'],
            'lr': 2 * self._config.LEARNING_RATE},
            {'params': [param for name, param in self.model.named_parameters() if name[-4:] != 'bias'],
            'lr': self._config.LEARNING_RATE, 'weight_decay': self._config.WEIGHT_DECAY}
            ], momentum=self._config.MOMENTUM, nesterov=True)
        imgs_train = glob.glob(os.path.join(self._config.FOLDER_WITH_IMAGE_DATA, "train", "imgs/*.png"))
        masks_train = glob.glob(os.path.join(self._config.FOLDER_WITH_IMAGE_DATA, "train", "masks/*.png"))
        imgs_val = glob.glob(os.path.join(self._config.FOLDER_WITH_IMAGE_DATA, "validate", "imgs/*.png"))
        masks_val = glob.glob(os.path.join(self._config.FOLDER_WITH_IMAGE_DATA, "validate", "masks/*.png"))
        imgs_train.sort()
        imgs_val.sort() 
        masks_train.sort()
        masks_val.sort()
        dataloader_train = DataLoaderCrop2D(img_files=imgs_train, mask_files=masks_train, 
                                           crop_size=(self._config.CROP_SIZE, self._config.CROP_SIZE), 
                                           stride=self._config.STRIDE, transform=self._config.AUGMENTATION)
        dataloader_val = DataLoaderCrop2D(img_files=imgs_val, mask_files=masks_val, 
                                        crop_size=(self._config.CROP_SIZE, self._config.CROP_SIZE), 
                                        stride=self._config.STRIDE, transform=self._config.VAL_AUGMENTATION)
        self.loader_train = DataLoader(dataloader_train, batch_size=self._config.BATCH_SIZE, shuffle=True, num_workers=0)
        # TODO: currently only validation with batch_size 1 is supported
        self.loader_val = DataLoader(dataloader_val, batch_size=1, shuffle=False, num_workers=0)

        if not os.path.exists(os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER)):
            os.makedirs(os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER))
        
    def _save_segmentation(self, prediction, count_map, img_path, mask_path, name):
        img_name = os.path.split(img_path)[1][:-4]
        prediction = prediction / count_map
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        plt.subplot(1, 2, 1)
        plt.imshow(prediction[0, :, :])
        plt.subplot(1, 2, 2)
        plt.imshow(prediction[1, :, :])
        plt.savefig(os.path.join(name, img_name + "_maps.png"), bbox_inches="tight")
        savemat(os.path.join(name, img_name + ".mat"), {"pred":prediction})
        prediction = np.argmax(prediction, axis=0)
        prediction_gt = vizualize_segmentation(np.array(gt > 0).astype(np.uint8), np.array(prediction > 0).astype(np.uint8))
        cv2.imwrite(os.path.join(name, img_name + "gt_vs_pred.png"), prediction_gt)
        cv2.imwrite(os.path.join(name, img_name + "_prediction.png"), map_palette(
            prediction, generate_palette(self._config.NUM_CLASSES)))        
        cv2.imwrite(os.path.join(name, img_name + "_gt.png"), map_palette(
            gt /255, generate_palette(self._config.NUM_CLASSES)))
        cv2.imwrite(os.path.join(name, img_name + "_img_vs_pred.png"), 
                show_segmentation_into_original_image(img, prediction))
        shutil.copy(img_path, os.path.join(name, img_name + ".png"))
        
    def save_model(self, epoch_number=0):
        now = datetime.now()
        epoch_str = "_epoch{}_".format(epoch_number)
        now = now.strftime("_%m-%d-%Y_%H:%M:%S_")
        filename = str(self.model) + epoch_str + now + str(self.average_meter_val) + ".pth"
        torch.save(self.model.state_dict(), os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER, filename))
        torch.save(self.optimizer.state_dict(), os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER, "opt_" + filename))
        
    def load_model(self, path_ckpt, opt_path_ckpt):
        self.model.load_state_dict(torch.load(path_ckpt))
        self.optimizer.load_state_dict(torch.load(opt_path_ckpt))
        print("Model loaded: {}".format(path_ckpt))

    def _load_weights_if_available(self):
        if len(self._config.CHECKPOINT) > 0:
            self.load_model(
                os.path.join(Configuration.OUTPUT, Configuration.OUTPUT_FOLDER, Configuration.CHECKPOINT),
                os.path.join(Configuration.OUTPUT, Configuration.OUTPUT_FOLDER, "opt_" + Configuration.CHECKPOINT))


if __name__ == "__main__":
    trainer = TrainModel(Configuration)
    trainer.train()