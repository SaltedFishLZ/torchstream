# Something-something V1 Dataset
# https://20bn.com/datasets/something-something/v1

__dataset__ = "Sth-sth-v1"
__style__ = "20BN"

from . import path

from .path import RAW_DATA_PATH, PRC_DATA_PATH, \
    JPG_DATA_PATH
from .label import __labels__, __sample_num_per_class__, __targets__
from .split import TrainsetFilter, ValsetFilter, TestsetFilter

__all__ = [
    "__dataset__", "__style__",
    "RAW_DATA_PATH", "PRC_DATA_PATH", "JPG_DATA_PATH",
    "__labels__", "__targets__", "__sample_num_per_class__",
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]