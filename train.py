from collections import namedtuple
from datetime import datetime
import glob
import os
import shutil
import sys

import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import available_models, Configuration, TickColonSegmentation
from helper_scripts.utils import check_and_make_dirs
from loaders.data_loader_mask_generic import DataLoaderCrop2D
from loaders.get_data_loader import get_data_loaders
from models.combine_net import CombineNet
from stats.stats_meter import StatsMeter
from utils.segmentation_vizualization import (
    generate_palette, map_palette, show_segmentation_into_original_image, vizualize_segmentation)
from visualization import visualizers


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
        self._writer = SummaryWriter()
        self._initialize()
        self._load_weights_if_available()
        self._visualizer = visualizers[self._config.VISUALIZER](self._config, self._writer)

    def train(self):
        for epoch in range(self._config.NUMBER_OF_EPOCHS):
            self.validate(-1)
            self.model.train()
            self.average_meter_train = StatsMeter(self._config.NUM_CLASSES)
            tqdm_loader = tqdm(enumerate(self.loader_train))
            for idx, data in tqdm_loader:
                tqdm_loader.set_description("Last IOU: {:.3f}".format(self.average_meter_train.last_iou))
                tqdm_loader.refresh()
                img, mask, indices, img_path, mask_path = data
                self.optimizer.zero_grad()
                if self._config.CUDA:
                    img = img.cuda()
                    mask = mask.cuda()

                output = self.model(img)
                prediction = torch.argmax(output, dim=1)
                loss = self.loss(output, mask)
                loss.backward()
                self.optimizer.step()
                self.average_meter_train.update(prediction.cpu().numpy(), mask.cpu().numpy(), loss.item())
                self._writer.add_scalar("Loss/train", loss.item(), idx + len(self.loader_train) * epoch)
<<<<<<< HEAD
                self._writer.add_scalar("Precision/train", torch.sum(prediction == mask) / (
                     prediction.shape[1] * prediction.shape[2]), idx + len(self.loader_train) * epoch)
=======
                self._writer.add_scalar("Precision/train", torch.sum(prediction == mask) / (prediction.shape[0] * prediction.shape[1]), idx + len(self.loader_train) * epoch)
>>>>>>> b190a59210f116028110c0bf9a353372b09d1ebd

            print("\n" + "-" * 50 + "\n")
            print(self.average_meter_train)
            print("\n" + "-" * 50 + "\n")
            if epoch % self._config.VALIDATION_FREQUENCY == 0:
<<<<<<< HEAD
                torch.cuda.empty_cache()
=======
>>>>>>> b190a59210f116028110c0bf9a353372b09d1ebd
                self.validate(epoch)
                self.model.train()
                self.save_model(epoch)

    def validate(self, epoch_num=0):
        self.model.eval()
        path_to_save = os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER, str(epoch_num) + "_epoch")
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
            
        self.average_meter_val = StatsMeter(self._config.NUM_CLASSES)
        img_shape = None
        CurrentlyOpened = namedtuple("CurrentlyOpened", ["img", "mask"])
        opened = CurrentlyOpened(None, None)
        count_map = None
        output_segmented = None

        for idx, data in tqdm(enumerate(self.loader_val)):
            img, mask, indices, img_path, mask_path = data

            if self._config.CUDA:
                img = img.cuda()
                mask = mask.cuda()

            if opened.img != img_path[0]:
                if count_map is not None and output_segmented is not None:
                    self._save_segmentation(
                        output_segmented.cpu().numpy(), count_map.cpu().numpy(), 
                        opened.img, opened.mask, path_to_save, idx=epoch_num * len(self.loader_val) + idx)
                opened = CurrentlyOpened(img_path[0], mask_path[0])
                img_shape = cv2.imread(opened.img, cv2.IMREAD_GRAYSCALE).shape
                output_segmented = torch.zeros((self._config.NUM_CLASSES, img_shape[0], img_shape[1])).cuda()
                count_map = torch.zeros(img_shape).cuda()
            
            output = self.model(img)
            prediction = torch.argmax(output, dim=1)
            loss = self.loss(output, mask)

            # TODO: print val stats...
            self.average_meter_val.update(prediction.cpu().numpy(), mask.cpu().numpy(), loss.item())
            self._writer.add_scalar("Loss/validation", loss.item(), epoch_num * len(self.loader_val) + idx)
<<<<<<< HEAD
            self._writer.add_scalar("Precision/validation", torch.sum(prediction == mask) / (
                prediction.shape[1] * prediction.shape[2]), epoch_num * len(self.loader_val) + idx)

            output_segmented[:, indices[0]: indices[2], indices[1]: indices[3]] += output[0, :, :, :].data
            count_map[indices[0]: indices[2], indices[1]: indices[3]] += 1
            torch.cuda.empty_cache()
=======
            self._writer.add_scalar("Precision/validation", torch.sum(prediction == mask) / (prediction.shape[0] * prediction.shape[1]), epoch_num * len(self.loader_val) + idx)

            output_segmented[:, indices[0]: indices[2], indices[1]: indices[3]] += output[0, :, :, :].data
            count_map[indices[0]: indices[2], indices[1]: indices[3]] += 1
>>>>>>> b190a59210f116028110c0bf9a353372b09d1ebd
        self._save_segmentation(
            output_segmented.cpu().numpy(), count_map.cpu().numpy(),
            opened.img, opened.mask, path_to_save, idx=epoch_num * len(self.loader_val) + idx)
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
        self.loader_train, self.loader_val = get_data_loaders(self._config)
        check_and_make_dirs(os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER))
       
    def _save_segmentation(self, prediction, count_map, img_path, mask_path, name, idx):
        self._visualizer.process_output(prediction, count_map, img_path, mask_path, name=name, idx=idx)

    def save_model(self, epoch_number=0):
        now = datetime.now()
        epoch_str = "_epoch{}_".format(epoch_number)
        now = now.strftime("_%m-%d-%Y_%H_%M_%S_")
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
                os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER, self._config.CHECKPOINT),
                os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER, "opt_" + self._config.CHECKPOINT))


if __name__ == "__main__":
    trainer = TrainModel(TickColonSegmentation())
    trainer.train()
