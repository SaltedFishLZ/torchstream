import os
import time
import pickle

import pandas

from ...utilities import touch_date
from .csv_parse import TRAINSET_DF, VALSET_DF, TESTSET_DF

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# ---------------------------------------------------------------- #
#               Main Classes (To Be Used Externally)               #        
# ---------------------------------------------------------------- #

class DatasetFilter(object):
    """
    This is an abstract class of common codes for different splits
    """
    def __init__(self, split="train"):
        # santity check
        assert split in ["train", "val", "test"], \
            "Invalid dataset split"

        self.split = split
        self.split_set = set()
        if "train" == self.split:
            self.split_df = TRAINSET_DF
        elif "val" == self.split:
            self.split_df = VALSET_DF
        else:
            self.split_df = TESTSET_DF

        set_file = os.path.join(DIR_PATH, \
            "something-something-v1.{}.set".format(split))
        ## find valid cache
        if (os.path.exists(set_file)
                and (touch_date(FILE_PATH) < touch_date(set_file))):
            print("Find valid set cache")
            fin = open(set_file, "rb")
            self.split_set = pickle.load(fin)
            fin.close()
        ## re-generate set file and dump it
        else:
            for idx, row in self.split_df.iterrows():
                video = row["video"]
                self.split_set.add(video)
            # TODO: consistency issue    
            fout = open(set_file, "wb")
            pickle.dump(self.split_set, fout)
            fout.close()

    def __call__(self, sample):
        if sample.name in self.split_set:
            return True
        return False


class TrainsetFilter(DatasetFilter):
    def __init__(self):
        super(TrainsetFilter, self).__init__(split="train")
        self.trainset = self.split_set

class TestsetFilter(DatasetFilter):
    def __init__(self):
        super(TestsetFilter, self).__init__(split="test")
        self.testset = self.split_set

class ValsetFilter(DatasetFilter):
    def __init__(self):
        super(ValsetFilter, self).__init__(split="val")
        self.valset = self.split_set

## Self Test Function
#  
#  Details
def test():
    """
    Self-test function
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
