# UCF101 Dataset
# https://www.crcv.ucf.edu/data/UCF101.php

from .data_path import raw_data_path, prc_data_path
from .label_map import label_map
from .data_split import __supported_splits__, for_train, for_val, for_test

__all__ = ["raw_data_path", "prc_data_path", "label_map",
        "__supported_splits__", "for_train", "for_val", "for_test"]
        