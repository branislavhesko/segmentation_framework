import numpy as np


class StatsMeter:

    def __init_(self, num_classes):
        self._number_of_images_passed = 0
        self._sum_of_losses = 0
        self._correctly_predicted_pixels = 0
        self._total_predicted_pixels = 0
        self._num_classes = num_classes
        self._ious = np.zeros(self._num_classes)
        self._dices = np.zeros(self._num_classes)

    def reset(self):
        self._sum_of_losses = 0
        self._number_of_images_passed = 0
        self._correctly_predicted_pixels = 0
        self._total_predicted_pixels = 0
        self._ious = np.zeros(self._num_classes)
        self._dices = np.zeros(self._num_classes)

    @property
    def mean_loss(self):
        return self._sum_of_losses / self._number_of_images_passed

    @property
    def accuracy(self):
        return self._correctly_predicted_pixels / self._total_predicted_pixels
    
    @property
    def mean_iou(self):
        return np.mean(self._ious)

    @property
    def mean_dice(self):
        return np.mean(self._dices)

    def update(self, prediction, ground_truth, loss):
        self._number_of_images_passed += 1
        self._sum_of_losses += loss
        self._correctly_predicted_pixels += np.sum(prediction == ground_truth)
        self._total_predicted_pixels += prediction.shape[0] * prediction.shape[1]

        for i in range(self._num_classes):
            iou = np.bitwise_and(prediction == i, ground_truth == i) / np.bitwise_or(prediction == i, ground_truth == i)
            self._ious[i] = ((self._ious[i] * (self._number_of_images_passed - 1)) + iou) / self._number_of_images_passed

    def __str__(self):
        desc = "NUM_CLASSES: {}, mean_loss: {}, accuracy: {}, mean_IOU: {}, mean_DICE: {}".format(
            self._num_classes, self.mean_loss, self.accuracy, self.mean_iou, self.mean_dice)
        return desc