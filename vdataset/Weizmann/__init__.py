# Weizmann Dataset
# http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html

from .data_path import raw_data_path, prc_data_path
from .label_map import label_map
from .data_split import for_train, for_val, for_test

__all__ = ["raw_data_path", "prc_data_path", "label_map", "for_train",
        "for_val", "for_test"]