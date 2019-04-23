# Weizmann Dataset
# http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html

# class list
__classes__ = [
    "bend",
    "jack",
    "jump",
    "pjump",
    "run",
    "side",
    "skip",
    "walk",
    "wave1",
    "wave2",
]

# sample number of each class
# if is a list, it is an estimation [min, max] (including)
__samples__ = {
    "bend"  :   9,
    "jack"  :   9,
    "jump"  :   9,
    "pjump" :   9,
    "run"   :   10,
    "side"  :   9,
    "skip"  :   10,
    "walk"  :   10,
    "wave1" :   9,
    "wave2" :   9,
}

from .data_path import raw_data_path, prc_data_path
from .label_map import label_map
from .data_split import for_train, for_val, for_test

__all__ = [
    "__classes__", "__samples__",
    "raw_data_path", "prc_data_path",
    "label_map", "for_train", "for_val", "for_test"
]

if __name__ == "__main__":
    print("common Data of Weizmann Dataset")
    print(__samples__)
