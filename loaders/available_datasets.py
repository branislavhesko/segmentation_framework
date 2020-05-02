from loaders.data_loader_mask_generic import DataLoaderCrop2D
from loaders.refugee_data_loader import RefugeeSegmentationDataset
from loaders.tick_data_loader import TickDataLoaderCrop2D


class AvailableDatasets:
    DATASETS = {
        "DataLoaderCrop2D": DataLoaderCrop2D,
        "TickDataLoaderCrop2D": TickDataLoaderCrop2D,
        "RefugeeDataset": RefugeeSegmentationDataset
    }
