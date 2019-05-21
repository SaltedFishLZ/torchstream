# UCF101 Dataset
# https://www.crcv.ucf.edu/data/UCF101.php

__dataset__ = "UCF101"
__style__ = "UCF101"

from .path import RAW_DATA_PATH, PRC_DATA_PATH, \
    AVI_DATA_PATH, JPG_DATA_PATH
from .label import __labels__, __sample_num_per_class__
from .split import TrainsetFilter, ValsetFilter, TestsetFilter

__all__ = [
    "__dataset__", "__style__",
    "RAW_DATA_PATH", "PRC_DATA_PATH",
    "AVI_DATA_PATH", "JPG_DATA_PATH",
    "__labels__", "__sample_num_per_class__",
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]

if __name__ == "__main__":
    print("Common Data of UCF101 Dataset")
    print(__sample_num_per_class__)
