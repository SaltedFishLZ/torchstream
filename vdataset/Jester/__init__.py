# Jester V1 Dataset
# https://20bn.com/datasets/jester/v1

__dataset__ = "Jester"
__style__ = "20BN"

from .path import raw_data_path, prc_data_path
from .label import __labels__, __sample_num_per_class__, __targets__
from .split import TrainsetFilter, ValsetFilter, TestsetFilter

__all__ = [
    "__dataset__", "__style__",
    "raw_data_path", "prc_data_path",
    "__labels__", "__targets__", "__sample_num_per_class__",
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]

if __name__ == "__main__":
    print("Common Data of Jester Dataset")
    print(__sample_num_per_class__)
