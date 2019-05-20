# Weizmann Dataset
# http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html

__dataset__ = "Weizmann"
__style__ = "UCF101"

from .path import raw_data_path, prc_data_path
from .label import __labels__, __sample_num_per_class__
from .split import TrainsetFilter, ValsetFilter, TestsetFilter

__all__ = [
    "__dataset__", "__style__",
    "raw_data_path", "prc_data_path",
    "__labels__", "__sample_num_per_class__",
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]
