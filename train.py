from datetime import datetime
import glob
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from config import available_models, Configuration
from loaders.data_loader_mask_generic import DataLoaderCrop2D
from models.combine_net import CombineNet
from stats.stats_meter import StatsMeter


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

    def train(self):
        for epoch in range(self._config.NUMBER_OF_EPOCHS):
            self.model.train()
            self.save_model()
            self.validate()
            for data in tqdm(self.loader_train):
                img, mask, indices, img = data
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

            print(self.average_meter_train)

            if epoch % self._config.VALIDATION_FREQUENCY == 0:
                self.validate()

    def validate(self):
        self.model.eval()
        self.average_meter_val = StatsMeter(self._config.NUM_CLASSES)
        opened_image = ""
        for data in tqdm(self.loader_val):
            img, mask, indices, img_index = data

            if self._config.CUDA:
                img = img.cuda()
                mask = mask.cuda()

            if opened_image != img_index[0]:
                opened_image = cv2.imread(img_index[0], cv2.IMREAD_GRAYSCALE)
                output_segmented = torch.zeros((self._config.NUM_CLASSES, opened_image.shape[0], opened_image.shape[1])).cuda()
                count_map = torch.zeros(opened_image.shape).cuda()
            
            output = self.model(img)
            prediction = torch.argmax(output, dim=1)
            loss = self.loss(output, mask)

            self.average_meter_val.update(prediction.cpu().numpy(), mask.cpu().numpy(), loss.item())

            output_segmented[:, indices[0]: indices[2], indices[1]: indices[3]] += output[0, :, :, :].data
            count_map[indices[0]: indices[2], indices[1]: indices[3]] += 1







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
                                        stride=self._config.STRIDE, transform=self._config.AUGMENTATION)
        self.loader_train = DataLoader(dataloader_train, batch_size=self._config.BATCH_SIZE, shuffle=True, num_workers=4)
        # TODO: currently only validation with batch_size 1 is supported
        self.loader_val = DataLoader(dataloader_val, batch_size=1, shuffle=False)

        if not os.path.exists(os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER)):
            os.makedirs(os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER))

    def save_model(self):
        now = datetime.now()
        now = now.strftime("_%m-%d-%Y_%H:%M:%S_")
        filename = str(self.model) + now + str(self.average_meter_val) + ".pth"
        torch.save(self.model.state_dict(), os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER, filename))
        torch.save(self.optimizer.state_dict(), os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER, "opt_" + filename))
        
    def load_model(self, path_ckpt):
        self.model.load_state_dict(torch.load(path_ckpt))
        self.optimizer.load_state_dict(torch.load("opt_" + path_ckpt))


if __name__ == "__main__":
    trainer = TrainModel(Configuration)

    trainer.train()