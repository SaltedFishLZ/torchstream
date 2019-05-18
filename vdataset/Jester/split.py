import os
import pandas

from .common import trainset_df, valset_df, testset_df


# ---------------------------------------------------------------- #
#               Main Classes (To Be Used Externally)               #        
# ---------------------------------------------------------------- #

class DatasetFilter(object):
    '''
    This is an abstract class of common codes for different splits
    '''
    def __init__(self, split="train"):
        # santity check
        assert split in ["train", "val", "test"], \
            "Invalid dataset split"

        # main stuff
        self.split = split
        self.split_set = set()
        if ("train" == self.split):
            self.split_df = trainset_df
        elif ("val" == self.split):
            self.split_df = valset_df
        else:
            self.split_df = testset_df

        for idx, row in self.split_df.iterrows():
            video = str(row["video"])
            self.split_set.add(video)

    def __call__(self, sample):
        if (sample.name in self.split_set):
            return True
        else:
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



if __name__ == "__main__":
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