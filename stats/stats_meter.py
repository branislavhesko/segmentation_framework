import numpy as np


class StatsMeter:

    def __init__(self, num_classes):
        self._sum_of_losses = 0
        self._correctly_predicted_pixels = 0
        self._total_predicted_pixels = 1e-5
        self._num_classes = num_classes
        self._ious = np.zeros(self._num_classes)
        self._dices = np.zeros(self._num_classes)
        self.last_iou = np.zeros(self._num_classes)
        self._number_of_images_passed = np.zeros(self._num_classes) + 1e-5

    def reset(self):
        self._sum_of_losses = 0
        self._number_of_images_passed = np.zeros(self._num_classes) + 1e-5
        self._correctly_predicted_pixels = 0
        self._total_predicted_pixels = 0
        self._ious = np.zeros(self._num_classes)
        self._dices = np.zeros(self._num_classes)
        self.last_iou = np.zeros(self._num_classes)

    @property
    def mean_loss(self):
        return self._sum_of_losses / np.amax(self._number_of_images_passed)

    @property
    def accuracy(self):
        return self._correctly_predicted_pixels / self._total_predicted_pixels
    
    @property
    def mean_iou(self):
        return np.mean(np.divide(self._ious, self._number_of_images_passed))

    @property
    def iou(self):
        return np.divide(self._ious, self._number_of_images_passed)

    @property
    def mean_dice(self):
        return np.mean(np.divide(self._dices, self._number_of_images_passed))

    def update(self, prediction, ground_truth, loss):
        self._sum_of_losses += loss
        self._correctly_predicted_pixels += np.sum(prediction == ground_truth)
        self._total_predicted_pixels += prediction.shape[-1] * prediction.shape[-2] * prediction.shape[0]
        for class_num in range(self._num_classes):
            if np.sum(np.bitwise_and(prediction == class_num, ground_truth == class_num)) == 0:
                continue
            iou = np.sum(np.bitwise_and(prediction == class_num, ground_truth == class_num)) / \
                  np.sum(np.bitwise_or(prediction == class_num, ground_truth == class_num))
            dice = 2 * np.sum(np.bitwise_and(prediction == class_num, ground_truth == class_num)) / \
                   (np.sum(prediction == class_num) + np.sum(ground_truth == class_num))
            self._ious[class_num] += iou
            self._dices[class_num] += dice
            self.last_iou[class_num] = iou
            self._number_of_images_passed[class_num] += 1

    def __str__(self):
        desc = "NUM_CLASSES{}_mean_loss{:.3f}_accuracy{:.3f}_mean_IOU{:.3f}_mean_DICE{:.3f}".format(
            self._num_classes, self.mean_loss, self.accuracy, np.mean(self.mean_iou), np.mean(self.mean_dice))
        return desc