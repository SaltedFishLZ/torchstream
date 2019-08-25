# Something-something V1 Dataset
# https://20bn.com/datasets/something-something/v1

__dataset__ = "Sth-sth-v1"
__layout__ = "20BN"

__all__ = [
    "__dataset__", "__layout__",
    "__LABELS__", "__SAMPLES_PER_LABEL__", "__ANNOTATIONS__",
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]
from . import path
__all__ += path.__all__
from .label import __LABELS__, __SAMPLES_PER_LABEL__, __ANNOTATIONS__
from .split import TrainsetFilter, ValsetFilter, TestsetFilter
from .path import *