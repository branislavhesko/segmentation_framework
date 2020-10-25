from collections import namedtuple
from datetime import datetime
import logging

import os
from time import time

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import available_models, Configuration, TickColonSegmentation, \
    RefugeeDiscSegmentationConfig, RefugeeCupSegmentationConfig, ThyroidConfig, IdridSegmentation
from helper_scripts.utils import check_and_make_dirs
from loaders.get_data_loader import get_data_loaders
from stats.stats_meter import StatsMeter
from utils.misc import configuration_to_json

from visualization import visualizers


logging.basicConfig(format='%(levelname)s: [%(asctime)s] [%(name)s:%(lineno)d-%(funcName)20s()] %(message)s',
                    level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S')


class TrainModel:

    def __init__(self, config: Configuration):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config = config
        self.device = "cuda" if self._config.CUDA and torch.cuda.is_available() else "cpu"
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
        self._visualizer = visualizers[self._config.VISUALIZER](self._config, writer=self._writer)
        self._logger.info(config.serialize())

    def train(self):
        loss, output, prediction = None, None, None
        for epoch in range(self._config.NUMBER_OF_EPOCHS):
            self.model.train()
            self.average_meter_train = StatsMeter(self._config.NUM_CLASSES)
            tqdm_loader = tqdm(self.loader_train)
            for idx, data in enumerate(tqdm_loader):
                start = time()
                img, mask, indices, img_path, mask_path = data
                self.optimizer.zero_grad()
                img = img.to(self.device)
                mask = mask.to(self.device)

                output = self.model(img)
                prediction = torch.argmax(output, dim=1)
                loss = self.loss(output, mask)
                loss.backward()
                self.optimizer.step()
                tqdm_loader.set_description("Last IOU: {:.3f}, INFERENCE time: {:.2f}".format(
                    self.average_meter_train.last_iou, time() - start))
                tqdm_loader.refresh()
                self.average_meter_train.update(prediction.cpu().numpy(), mask.cpu().numpy(), loss.item())
                self._writer.add_scalar("Loss/train", loss.item(), idx + len(self.loader_train) * epoch)
                self._writer.add_scalar("Precision/train", torch.sum(prediction == mask).float() / (
                        prediction.shape[0] * prediction.shape[1]), idx + len(self.loader_train) * epoch)

            del loss, output,  prediction
            print("\n" + "-" * 50 + "\n")
            print(self.average_meter_train)
            print("\n" + "-" * 50 + "\n")
            if epoch % self._config.VALIDATION_FREQUENCY == 0:
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                self.validate(epoch)
                self.model.train()
                self.save_model(epoch)

    @torch.no_grad()
    def validate(self, epoch_num=0):
        self.model.eval()
        path_to_save = os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER, str(epoch_num) + "_epoch")
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        configuration_to_json(self._config, os.path.join(path_to_save, "configuration.json"))
        self.average_meter_val = StatsMeter(self._config.NUM_CLASSES)
        CurrentlyOpened = namedtuple("CurrentlyOpened", ["img", "mask"])
        opened = CurrentlyOpened(None, None)
        count_map = None
        output_segmented = None
        image_id = 0
        for idx, data in enumerate(tqdm(self.loader_val)):
            img, mask, indices, img_path, mask_path = data

            img = img.to(self.device)
            mask = mask.to(self.device)

            if opened.img != img_path[0]:
                if count_map is not None and output_segmented is not None:
                    image_id += 1
                    if image_id % self._config.SAVE_FREQUENCY == 0:
                        self._save_segmentation(
                            output_segmented.cpu().numpy(), count_map.cpu().numpy(),
                            opened.img, opened.mask, path_to_save, idx=idx, epoch=epoch_num)
                opened = CurrentlyOpened(img_path[0], mask_path[0])
                img_shape = cv2.imread(opened.img, cv2.IMREAD_GRAYSCALE).shape
                output_segmented = torch.zeros((
                    self._config.NUM_CLASSES, img_shape[0], img_shape[1]), device=self.device)
                count_map = torch.zeros(img_shape, device=self.device)
            output = self.model(img)

            prediction = torch.argmax(output, dim=1)
            loss = self.loss(output, mask)

            # TODO: print val stats...
            self.average_meter_val.update(prediction.cpu().numpy(), mask.cpu().numpy(), loss.item())
            self._writer.add_scalar("Loss/validation", loss.item(), epoch_num * len(self.loader_val) + idx)
            self._writer.add_scalar("Precision/validation", torch.sum(prediction == mask).float() / (
                    prediction.shape[0] * prediction.shape[1]), epoch_num * len(self.loader_val) + idx)

            output_segmented[:, indices[0]: indices[2], indices[1]: indices[3]] += output[0, :, :, :].data
            count_map[indices[0]: indices[2], indices[1]: indices[3]] += 1
            torch.cuda.empty_cache()
        self._save_segmentation(output_segmented.cpu().numpy(), count_map.cpu().numpy(),
                                opened.img, opened.mask, path_to_save, idx=idx, epoch=epoch_num)
        print("\n" + "-" * 50 + "\n")
        print(self.average_meter_val)
        print("\n" + "-" * 50 + "\n")
        torch.cuda.empty_cache()
        del loss, output_segmented, prediction, count_map

    def _initialize(self):
        self.model = available_models[self._config.MODEL](self._config.NUM_CLASSES).to(self.device)
        print(self.model)
        self.loss = self._config.LOSS(config=self._config)
        self.optimizer = self._config.OPTIMALIZER([
            {'params': [param for name, param in self.model.named_parameters() if name[-4:] == 'bias'],
            'lr': 2 * self._config.LEARNING_RATE},
            {'params': [param for name, param in self.model.named_parameters() if name[-4:] != 'bias'],
            'lr': self._config.LEARNING_RATE, 'weight_decay': self._config.WEIGHT_DECAY}
            ], momentum=self._config.MOMENTUM, nesterov=True)
        self.loader_train, self.loader_val = get_data_loaders(self._config)
        check_and_make_dirs(os.path.join(self._config.OUTPUT, self._config.OUTPUT_FOLDER))
       
    def _save_segmentation(self, prediction, count_map, img_path, mask_path, name, idx, epoch):
        self._visualizer.process_output(prediction, count_map, img_path, mask_path, name=name, idx=idx, epoch=epoch)

    def save_model(self, epoch_number=0):
        now = datetime.now()
        epoch_str = "_epoch{}_".format(epoch_number)
        now = now.strftime("_%m-%d-%Y_%H_%M_%S_")
        filename = str(self._config.MODEL) + epoch_str + now + str(self.average_meter_val) + ".pth"
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
    trainer = TrainModel(IdridSegmentation())
    trainer.train()
