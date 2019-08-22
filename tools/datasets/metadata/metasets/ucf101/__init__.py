# UCF101 Dataset
# https://www.crcv.ucf.edu/data/UCF101.php
__dataset__ = "UCF101"
__layout__ = "UCF101"
__all__ = [
    "__dataset__", "__layout__",
    "__LABELS__", "__SAMPLES_PER_LABEL__",
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]
from . import path
__all__ += path.__all__
from .label import __LABELS__, __SAMPLES_PER_LABEL__
from .split import TrainsetFilter, ValsetFilter, TestsetFilter
from .path import *



if __name__ == "__main__":
    print("Common Data of UCF101 Dataset")
    print(__SAMPLES_PER_LABEL__)
