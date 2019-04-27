import os

__supported_splits__ = ["1", "2", "3"]

dir_path = os.path.dirname(os.path.realpath(__file__))

class TrainsetFilter(object):

    def __init__(self, split="1"):
        assert (split in __supported_splits__), "Unsupported split"
        self.train_set = set()
        list_file = "trainlist0" + split + ".txt"
        list_file = os.path.join(dir_path, list_file)   
        f = open(list_file, "r")
        for _line in f:
            _rel_path = _line.split('\n')[0]     # remove \n
            _rel_path = _rel_path.split(' ')[0]  # remove class id
            _rel_path = _rel_path.split('.')[0]  # remove file extension
            self.train_set.add(_rel_path)
        f.close()

    def __call__(self, sample):
        if ("{}/{}".format(sample.lbl, sample.name) 
            in self.train_set):
            return True
        else:
            return False

class TestsetFilter(object):

    def __init__(self, split="1"):
        assert (split in __supported_splits__), "Unsupported split"
        self.test_set = set()
        list_file = "testlist0" + split + ".txt"
        list_file = os.path.join(dir_path, list_file)   
        f = open(list_file, "r")
        for _line in f:
            _rel_path = _line.split('\n')[0]    # remove \n
            _rel_path = _rel_path.split(' ')[0] # remove class id
            _rel_path = _rel_path.split('.')[0] # remove file extension
            self.test_set.add(_rel_path)
        f.close()

    def __call__(self, sample):
        if ("{}/{}".format(sample.lbl, sample.name) 
            in self.test_set):
            return True
        else:
            return False

# set the same as test set
class ValsetFilter(object):
    def __init__(self, split="1"):
        self.TestsetFilter = TestsetFilter(split=split)
    def __call__(self, sample):
        return(self.TestsetFilter(sample))


if __name__ == "__main__":
    # self-test
    train_filter = TrainsetFilter()
    print(train_filter.train_set)
