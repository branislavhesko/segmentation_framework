from matplotlib import pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm
from config import available_models, Configuration, TickColonSegmentation, RefugeeDiscSegmentationConfig, RefugeeCupSegmentationConfig
from loaders.subimage_info_holder import ImageLoader


class Predictor:

    def __init__(self, config: Configuration, cuda: bool = False):
        self._config = config
        self.model = None
        self._cuda = cuda
        self._transform = self._config.VAL_AUGMENTATION
        weights_path = os.path.join("..", self._config.OUTPUT, self._config.OUTPUT_FOLDER, self._config.CHECKPOINT)
        assert os.path.exists(weights_path)
        self._load_model(weights_path)
        self._image_loader = ImageLoader((self._config.CROP_SIZE, self._config.CROP_SIZE), stride=0.2)

    def predict(self, image):
        assert len(image.shape) == 3, "Image should be in RGB format for now."
        self._image_loader.set_image_and_mask(image, image[:, :, 0])
        indices = self._image_loader.get_indices()
        count_map = np.zeros(image.shape[:2]).astype(np.float32)
        prediction = torch.zeros((self._config.NUM_CLASSES, *image.shape[:2]))
        if self._cuda:
            prediction = prediction.cuda()
        for idx in tqdm(indices):
            input_slice = np.copy(image[idx[0]: idx[2], idx[1]: idx[3], :])
            input_slice = self._transform(input_slice, input_slice[:, :, 0])[0]
            if self._cuda:
                input_slice = input_slice.cuda()
            with torch.no_grad():
                output = self.model(input_slice.unsqueeze(dim=0))
            prediction[:, idx[0]: idx[2], idx[1]: idx[3]] += output[0, :, :, :].data
            count_map[idx[0]: idx[2], idx[1]: idx[3]] += 1.

        prediction = prediction.cpu().numpy() / count_map
        return np.argmax(prediction, axis=0)

    def _load_model(self, weights_path: str):
        self.model = available_models[self._config.MODEL](self._config.NUM_CLASSES)
        if self._cuda:
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()


if __name__ == "__main__":
    import cv2
    from PIL import Image
    from matplotlib import pyplot as plt
    import glob
    images = glob.glob(os.path.join("/home/brani/STORAGE/DATA/refugee/test/*.jpg"))
    predictor = Predictor(RefugeeDiscSegmentationConfig(), cuda=True)
    os.makedirs("./data/output/", exist_ok=True)
    for image in images:
        base_name = os.path.splitext(os.path.basename(image))[0]
        if os.path.exists(os.path.join("./data/output/", base_name + ".png")):
            continue
        print("Processing image: {}".format(image))

        img = np.array(cv2.imread(image, cv2.IMREAD_COLOR))
        img_shape = img.shape[:2]
        img = cv2.resize(img, (1024, 1024))
        img = img / 255.
        mask = predictor.predict(img)
        mask = cv2.resize(mask, img_shape[::-1], interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join("./data/output/", base_name + ".png"), mask * 255)
