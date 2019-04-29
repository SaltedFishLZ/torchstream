# HMDB51 Dataset

__dataset__ = "HMDB51"
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

if __name__ == "__main__":
    print("Common Data of HMDB51 Dataset")
    print(__sample_num_per_class__)
