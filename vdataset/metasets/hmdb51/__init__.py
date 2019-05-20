# HMDB51 Dataset

__dataset__ = "HMDB51"
__style__ = "UCF101"

from .path import RAW_DATA_PATH, PRC_DATA_PATH
from .label import __labels__, __sample_num_per_class__
from .split import TrainsetFilter, ValsetFilter, TestsetFilter

__all__ = [
    "__dataset__", "__style__",
    "raw_data_path", "prc_data_path",
    "__labels__", "__sample_num_per_class__",
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]

