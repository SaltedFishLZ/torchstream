"""
Split filters
"""
__all__ = [
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]

import os

from . import __config__
from .label import __LABELS__
from .__support__ import __SUPPORTED_SPLIT_OPTS__

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)
LIST_PATH = os.path.join(DIR_PATH, "split_lists")



# ---------------------------------------------------------------- #
#     Building Training/Test/Validation Sets for Split Options     #
# ---------------------------------------------------------------- #

# a list of training/test sets of different split options
TRAINSETS = dict()
TESTSETS = dict()

for _split_option in __SUPPORTED_SPLIT_OPTS__:
    # create temporary sets for each split_option
    trainset = set()
    testset = set()

    for _label in __LABELS__:
        _file = "{}_test_split{}.txt".format(_label, _split_option)
        _file = os.path.join(LIST_PATH, _file)
        f = open(_file, "r")
        for _line in f:
            text = _line.split('\n')[0] # remove \n
            text = text.split(' ')      # split name and type
            _name = text[0]
            _name = _name.split('.')[0]
            _rec = '_'.join([_label, _name])
            # NOTE: Read split_readme.txt for more details. This is
            # HMDB official annotation:
            # http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/split_readme.txt
            if (text[1] == '1'):
                trainset.add(_rec)
            elif (text[1] == '2'):
                testset.add(_rec)
        f.close()

    TRAINSETS[_split_option] = trainset
    TESTSETS[_split_option] = testset



# ---------------------------------------------------------------- #
#               Main Classes (To Be Used Externally)               #        
# ---------------------------------------------------------------- #

class DatasetFilter(object):
    """An abstract class of common codes of different split filters
    """
    def __init__(self, split="train", split_option="1"):
        assert split_option in __SUPPORTED_SPLIT_OPTS__, \
            "Unsupported split option"
        self.split = split
        self.split_option = split_option

    def __call__(self, sample):
        name = sample.name
        label = sample.label
        rec = '_'.join([label, name])
        if "train" == self.split:
            return rec in TRAINSETS[self.split_option]
        if "test" == self.split:
            return rec in TESTSETS[self.split_option]
        if __config__.__STRICT__:
            raise NotImplementedError
        return False

class TrainsetFilter(DatasetFilter):
    """Wrapper: filter for training set
    """
    def __init__(self, split_option="1"):
        super(TrainsetFilter, self).__init__(split="train", split_option="1")

class TestsetFilter(DatasetFilter):
    """Wrapper: filter for testing set
    """
    def __init__(self, split_option="1"):
        super(TestsetFilter, self).__init__(split="test", split_option="1")

class ValsetFilter(DatasetFilter):
    """Wrapper: filter for validation set, the same as test set
    """
    def __init__(self, split_option="1"):
        # set the same as test set
        super(ValsetFilter, self).__init__(split="test", split_option="1")




if __name__ == "__main__":
    print("[Split 1]")
    print("Train - {}".format(len(TRAINSETS["1"])))
    print("Test - {}".format(len(TESTSETS["1"])))

    print("[Split 2]")
    print("Train - {}".format(len(TRAINSETS["2"])))
    print("Test - {}".format(len(TESTSETS["2"])))

    print("[Split 3]")
    print("Train - {}".format(len(TRAINSETS["3"])))
    print("Test - {}".format(len(TESTSETS["3"])))
