"""
Split filters
"""
__all__ = [
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]

import os
import time
import pickle
import logging

from . import __config__
from .jsonparse import TRAINSET_JLIST, VALSET_JLIST, TESTSET_JLIST
from ....utils.filesys import touch_date

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# ---------------------------------------------------------------- #
#                  Configuring Python Logger                       #
# ---------------------------------------------------------------- #

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
if __config__.__VERY_VERY_VERBOSE__:
    logger.setLevel(logging.INFO)
elif __config__.__VERY_VERBOSE__:
    logger.setLevel(logging.WARNING)
elif __config__.__VERBOSE__:
    logger.setLevel(logging.ERROR)
else:
    logger.setLevel(logging.CRITICAL)

# ---------------------------------------------------------------- #
#               Main Classes (To Be Used Externally)               #        
# ---------------------------------------------------------------- #

class DatasetFilter(object):
    """An abstract class of common codes of different split filters
    """
    def __init__(self, split="train"):
        # santity check
        assert split in ["train", "val", "test"], \
            "Invalid dataset split"

        self.split = split
        self.split_set = set()


        set_file = os.path.join(DIR_PATH, \
            "something-something-v2.{}.set".format(split))
        
        ## find valid cache
        if (os.path.exists(set_file)
                and (touch_date(FILE_PATH) < touch_date(set_file))):
            logger.info("Find valid {} set cache".format(split))
            fin = open(set_file, "rb")
            self.split_set = pickle.load(fin)
            fin.close()
        ## re-generate set file and dump it
        else:
            logger.info("Generating {} set info".format(split))

            if "train" == self.split:
                self.split_jlist = TRAINSET_JLIST
            elif "val" == self.split:
                self.split_jlist = VALSET_JLIST
            else:
                self.split_jlist = TESTSET_JLIST

            for _jdict in self.split_jlist:
                video = _jdict["id"]
                self.split_set.add(str(video))
            ## TODO: wrtie failure check
            fout = open(set_file, "wb")
            pickle.dump(self.split_set, fout)
            fout.close()

    def __call__(self, sample):
        if sample.name in self.split_set:
            return True
        return False

class TrainsetFilter(DatasetFilter):
    """Wrapper: filter for training set
    """
    def __init__(self):
        super(TrainsetFilter, self).__init__(split="train")
        self.trainset = self.split_set

class TestsetFilter(DatasetFilter):
    """Wrapper: filter for testing set
    """
    def __init__(self):
        super(TestsetFilter, self).__init__(split="test")
        self.testset = self.split_set

class ValsetFilter(DatasetFilter):
    """Wrapper: filter for validation set
    """
    def __init__(self):
        # the same as test set
        super(ValsetFilter, self).__init__(split="val")
        self.valset = self.split_set




## Self Test Function
def test():
    """Self-test function
    """
    st_time = time.time()
    # self-test
    print("Training Set")
    train_filter = TrainsetFilter()
    print(len(train_filter.trainset))
    # self-test
    print("Test Set")
    test_filter = TestsetFilter()
    print(len(test_filter.testset))
    # self-test
    print("Val Set")
    val_filter = ValsetFilter()
    print(len(val_filter.valset))

    ed_time = time.time()
    print("Total Time", ed_time - st_time)


if __name__ == "__main__":
    test()
