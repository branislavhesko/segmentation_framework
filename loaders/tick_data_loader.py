from loaders.data_loader_mask_generic import DataLoaderCrop2D


class TickDataLoaderCrop2D(DataLoaderCrop2D):
    MASK_CATEGORIES = ((0, 200, 200))
        
    def __init__(self, img_files, mask_files=(), crop_size=(512, 512),
                 stride=0.1, transform=lambda x: x):
        super().__init__(img_files, mask_files, crop_size,
                 stride, transform)    
        
    def __getitem__(self, index):
        info = self.sub_image_info_holder.get_info_at_index(index)
        img = np.array(cv2.imread(info.img, cv2.IMREAD_COLOR)).astype(np.float32)
        img = img / 255.
        mask = cv2.imread(info.mask, cv2.IMREAD_COLOR)
        mask[mask > 0] = 1
        data = (*self._transform(*self._crop_image_and_mask(img, mask, info)),
                info.slice, info.img, info.mask)
        return data