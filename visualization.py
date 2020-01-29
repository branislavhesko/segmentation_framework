import os
import shutil
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils.segmentation_vizualization import (
    generate_palette, map_palette, show_segmentation_into_original_image,
    vizualize_segmentation)


class VisualizationInterface(metaclass=ABCMeta):

    def __init__(self, config, **kwargs):
        self._config = config
    
    @abstractmethod
    def visualize_mask(self):
        pass
    
    @abstractmethod
    def process_output(self, prediction, count_map, img_path, mask_path, name):
        pass

    def store_prediction(self, prediction):
        fig, axs = plt.subplots(1, prediction.shape[0], figsize=(10, 3), dpi=100)
        for idx, ax in enumerate(axs):    
            ax.imshow(prediction[idx, :, :])
        return fig

class VisualizationSaveImages(VisualizationInterface):
    def visualize_mask():
        pass
    
    def process_output(self, prediction, count_map, img_path, mask_path, name, **kwargs):
        img_name = os.path.split(img_path)[1][:-4]
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        prediction = prediction / count_map
        fig = self.store_prediction(prediction)
        fig.savefig(os.path.join(name, img_name + "_maps.png"), bbox_inches="tight")
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


class VisualizationTensorboard(VisualizationInterface):

    def __init__(self, config, writer):
        super().__init__(config)
        self._writer: SummaryWriter = writer

    def visualize_mask():
        pass
    
    def process_output(self, prediction, count_map, img_path, mask_path, idx, **kwargs):
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        prediction = prediction / count_map
        fig = self.store_prediction(prediction)
        self._writer.add_figure("Prediction/maps", fig, global_step=idx)
        prediction = np.argmax(prediction, axis=0)
        prediction_gt = vizualize_segmentation(np.array(gt > 0).astype(np.uint8), np.array(prediction > 0).astype(np.uint8))
        self._writer.add_image("Prediction/colored", torch.from_numpy(map_palette(prediction, generate_palette(
            self._config.NUM_CLASSES))).permute([2, 0, 1]), global_step=idx)
        self._writer.add_image("PredictionVsGroundTruth/colored", torch.from_numpy(prediction_gt).permute([2, 0, 1]), global_step=idx)
        self._writer.add_image("Ground_truth/colored", torch.from_numpy(map_palette(
            gt > 0, generate_palette(self._config.NUM_CLASSES))).permute([2, 0, 1]), global_step=idx)
        self._writer.add_image("OriginalImage", torch.from_numpy(img).permute([2, 0, 1]), global_step=idx)
        self._writer.add_image("OriginalImageAndPrediction", torch.from_numpy(
            show_segmentation_into_original_image(img, prediction)).permute([2, 0, 1]), global_step=idx)


visualizers = {
    "VisualizationTensorboard": VisualizationTensorboard,
    "VisualizationSaveImages": VisualizationSaveImages
}