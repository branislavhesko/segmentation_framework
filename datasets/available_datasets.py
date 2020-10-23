from datasets.idrid_dataset import IdridDataset
from datasets.refugee_dataset import RefugeeSegmentationDataset
from datasets.thyroid_dataset import ThyroidDataset
from datasets.tick_colon_dataset import TickDataLoaderCrop2D
from loaders.data_loader_mask_generic import DataLoaderCrop2D
from loaders.smart_random_data_loader import SmartRandomDataLoader


class AvailableDatasets:
    DATASETS = {
        "DataLoaderCrop2D": DataLoaderCrop2D,
        "IdridDataset": IdridDataset,
        "TickDataLoaderCrop2D": TickDataLoaderCrop2D,
        "RefugeeDataset": RefugeeSegmentationDataset,
        "ThyroidDataset": ThyroidDataset,
        "SmartRandomDataLoader": SmartRandomDataLoader
    }
