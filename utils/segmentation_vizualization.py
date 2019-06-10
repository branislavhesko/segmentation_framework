import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio


def vizualize_segmentation(img1, img2):
    result_img = np.zeros((*img1.shape, 3))
    print(result_img.shape)
    intersetcion = cv2.bitwise_and(img1, img2)
    img1[intersetcion > 0] = 0
    img2[intersetcion > 0] = 0
    result_img[intersetcion > 0, 1] = 255
    result_img[img1 > 0, 0] = 255
    result_img[img2 > 0, 2] = 255
    return result_img.astype(np.uint8)


def store_predictions(filename, matrix):
    sio.savemat(filename, matrix)


def store_vizualization(filename, img):
    plt.figure(0, figsize=(16, 9), dpi=100)
    plt.imshow(img)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def show_segmentation_into_original_image(img, segmented_img):
    img[segmented_img > 0, 0] = 0
    img[segmented_img > 0, 2] = 0
    return img


def show_segmentations_into_original_image(img, segmented_img):
    colormaps = [[255, 0, 0],
                 [0, 255, 0],
                 [0, 0, 255],
                 [255, 0, 255],
                 [0, 255, 255],
                 ]

    for index, colormap in enumerate(colormaps):
        if colormap[0] == 0:
            img[segmented_img == index+1, 0] = 0
        if colormap[1] == 0: 
            img[segmented_img == index+1, 1] = 0
        if colormap[2] == 0:
            img[segmented_img == index+1, 2] = 0

    return img


def merge_images(img1, img2, mask):
    for channel in range(img1.shape[2]):
        img1[mask > 0, channel] = 0

    return img1 + img2

def generate_palette(num_classes):
    palette = [(0, 0, 0)]
    for i in range(num_classes - 1):
        palette.append((i, num_classes // 2 + i // 2, 255 - i))
    return palette

def map_palette(image, palette):
    img_colored = np.zeros((*image.shape, 3))
    for i in range(len(palette)):
        img_colored[image == i] = palette[i]

    return img_colored