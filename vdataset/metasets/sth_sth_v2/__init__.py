# Something-something V2 Dataset
# https://20bn.com/datasets/something-something/v2

__dataset__ = "Sth-sth-v2"
__style__ = "20BN"

from .path import RAW_DATA_PATH, PRC_DATA_PATH
from .label import __labels__, __sample_num_per_class__, __targets__
from .split import TrainsetFilter, ValsetFilter, TestsetFilter

__all__ = [
    "__dataset__", "__style__",
    "RAW_DATA_PATH", "PRC_DATA_PATH",
    "__labels__", "__targets__", "__sample_num_per_class__",
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]