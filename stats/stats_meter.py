import numpy as np


class StatsMeter:

    def __init__(self, num_classes):
        self._number_of_images_passed = 1e-5
        self._sum_of_losses = 0
        self._correctly_predicted_pixels = 0
        self._total_predicted_pixels = 1e-5
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
        if self._number_of_images_passed == 0:
            return 0
        return np.mean(self._ious / self._number_of_images_passed)

    @property
    def mean_dice(self):
        if self._number_of_images_passed == 0:
            return 0
        return np.mean(self._dices / self._number_of_images_passed)

    def update(self, prediction, ground_truth, loss):
        self._number_of_images_passed += 1
        self._sum_of_losses += loss
        self._correctly_predicted_pixels += np.sum(prediction == ground_truth)
        self._total_predicted_pixels += prediction.shape[-1] * prediction.shape[-2]
        for batch_num in range(prediction.shape[0]):
            for class_num in range(self._num_classes):
                iou = np.sum(np.bitwise_and(prediction[batch_num, :, :] == class_num, ground_truth[
                    batch_num, :, :] == class_num)) / np.sum(np.bitwise_or(prediction[
                        batch_num, :, :] == class_num, ground_truth[
                            batch_num, :, :] == class_num))
                dice = 2 * np.sum(np.bitwise_and(prediction[batch_num, :, :] == class_num, ground_truth[
                    batch_num, :, :] == class_num)) / (np.sum(prediction[
                        batch_num, :, :] == class_num) + np.sum(ground_truth[batch_num, :, :] == class_num))
                if np.isnan(iou) or np.isnan(dice) or np.isinf(dice) or np.isinf(iou):
                    continue
                self._ious[class_num] += iou
                self._dices[class_num] += dice

    def __str__(self):
        desc = "NUM_CLASSES:{}_mean_loss:{}_accuracy:{}_mean_IOU:{}_mean_DICE:{}".format(
            self._num_classes, self.mean_loss, self.accuracy, self.mean_iou, self.mean_dice)
        return desc