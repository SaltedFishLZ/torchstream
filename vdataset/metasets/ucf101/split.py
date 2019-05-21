import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Split Option is a feature of UCF101
# UCF101 is not so large, so the authors give 3 split options for Train/Test
# dataset split. Different official split options are stored as 
# '<test/train>list??.txt'
__supported_split_options__ = ["1", "2", "3"]



class DatasetFilter(object):
    '''
    This is an abstract class of common codes for different splits
    '''
    def __init__(self, split="train", split_option="1"):
        
        # santity check
        assert (split_option in __supported_split_options__), \
            "Unsupported split option"
        
        self.split = split
        self.split_option = split_option

        list_file = self.split + "list0" + split_option + ".txt"
        list_file = os.path.join(DIR_PATH, list_file) 
        
        # main stuff
        self.split_set = set()
  
        f = open(list_file, "r")
        for _line in f:
            _rel_path = _line.split('\n')[0]     # remove \n
            _rel_path = _rel_path.split(' ')[0]  # remove class id
            _rel_path = _rel_path.split('.')[0]  # remove file extension
            self.split_set.add(_rel_path)
        f.close()

    def __call__(self, sample):
        '''
        NOTE
        Here, we require the input `sample` to be an instance of the 
        `vdataset.metadata.Sample` type.
        '''
        if ("{}/{}".format(sample.lbl, sample.name) in self.split_set):
            return True
        else:
            return False



class TrainsetFilter(DatasetFilter):
    def __init__(self, split_option="1"):
        super(TrainsetFilter, self).__init__(split="train", split_option="1")
        self.trainset = self.split_set

class TestsetFilter(DatasetFilter):
    def __init__(self, split_option="1"):
        super(TestsetFilter, self).__init__(split="test", split_option="1")
        self.testset = self.split_set

class ValsetFilter(DatasetFilter):
    def __init__(self, split_option="1"):
        # the same as test set
        super(ValsetFilter, self).__init__(split="test", split_option="1")
        self.valset = self.split_set


if __name__ == "__main__":
    # self-test
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