"""
Split filters
"""
__all__ = [
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]

import os

from . import __config__
from .__support__ import __SUPPORTED_SPLIT_OPTS__

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

class DatasetFilter(object):
    """An abstract class of common codes of different split filters
    """
    def __init__(self, split, split_option="1"):
        
        # santity check
        assert (split_option in __SUPPORTED_SPLIT_OPTS__), \
            "Unsupported split option"
        
        self.split = split
        self.split_option = split_option

        list_file = self.split + "list0" + split_option + ".txt"
        list_file = os.path.join(DIR_PATH, list_file) 
        
        self.split_set = set()
        fin = open(list_file, "r")
        for _line in fin:
            _rel_path = _line.split('\n')[0]     # remove \n
            _rel_path = _rel_path.split(' ')[0]  # remove class id
            _rel_path = _rel_path.split('.')[0]  # remove file extension
            self.split_set.add(_rel_path)
        fin.close()

    def __call__(self, sample):
        """
        NOTE:
        Here, we require the input `sample` to be a Sample type.
        """
        return "{}/{}".format(sample.lbl, sample.name) in self.split_set


class TrainsetFilter(DatasetFilter):
    """Wrapper: filter for training set
    """
    def __init__(self, split_option="1"):
        super(TrainsetFilter, self).__init__(split="train", split_option="1")
        self.trainset = self.split_set

class TestsetFilter(DatasetFilter):
    """Wrapper: filter for testing set
    """
    def __init__(self, split_option="1"):
        super(TestsetFilter, self).__init__(split="test", split_option="1")
        self.testset = self.split_set

class ValsetFilter(DatasetFilter):
    """Wrapper: filter for validation set, the same as test set
    """
    def __init__(self, split_option="1"):
        # the same as test set
        super(ValsetFilter, self).__init__(split="test", split_option="1")
        self.valset = self.split_set



def test():
    """
    Self test
    """
    print("Training Set")
    train_filter = TrainsetFilter()
    print(len(train_filter.trainset))
    # self-test
    print("Test Set")
    test_filter = TestsetFilter(split_option="3")
    print(len(test_filter.testset))
    # self-test
    print("Val Set")
    val_filter = ValsetFilter(split_option="3")
    print(len(val_filter.valset))    


if __name__ == "__main__":
    test()