# HMDB51 Dataset

__dataset__ = "HMDB51"
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
    print("Common Data of HMDB51 Dataset")
    print(__SAMPLES_PER_LABEL__)
