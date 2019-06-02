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
        return np.mean(self._ious)

    @property
    def mean_dice(self):
        return np.mean(self._dices)

    def update(self, prediction, ground_truth, loss):
        self._number_of_images_passed += 1
        self._sum_of_losses += loss
        self._correctly_predicted_pixels += np.sum(prediction == ground_truth)
        self._total_predicted_pixels += prediction.shape[-1] * prediction.shape[-2]
        for batch_num in range(prediction.shape[0]):
            for class_num in range(self._num_classes):
                iou = np.bitwise_and(prediction[batch_num, class_num, :, :] == class_num, ground_truth[
                    batch_num, class_num, :, :] == class_num) / np.bitwise_or(prediction[
                        batch_num, class_num, :, :] == class_num, ground_truth[
                            batch_num, class_num, :, :] == class_num)
                #TODO: finish
                self._ious[class_num] = ((self._ious[class_num] * (self._number_of_images_passed - 1)) + iou) / self._number_of_images_passed

    def __str__(self):
        desc = "NUM_CLASSES:{}_mean_loss:{}_accuracy:{}_mean_IOU:{}_mean_DICE:{}".format(
            self._num_classes, self.mean_loss, self.accuracy, self.mean_iou, self.mean_dice)
        return desc