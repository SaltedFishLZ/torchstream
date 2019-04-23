import os

__supported_splits__ = ["1", "2", "3"]

dir_path = os.path.dirname(os.path.realpath(__file__))

class for_train(object):

    def __init__(self, split="1"):
        assert (split in __supported_splits__), "Unsupported split"
        self.train_set = set()
        list_file = "trainlist0" + split + ".txt"
        list_file = os.path.join(dir_path, list_file)   
        f = open(list_file, "r")
        for _line in f:
            _rel_path = _line.split('\n')[0]
            self.train_set.add(_rel_path)
        f.close()

    def __call__(self, sample):
        if ("{}/{}.{}".format(sample.lbl, sample.name, sample.ext) 
            in self.train_set):
            return True
        else:
            return False

class for_test(object):

    def __init__(self, split="1"):
        assert (split in __supported_splits__), "Unsupported split"
        self.test_set = set()
        list_file = "testlist0" + split + ".txt"
        list_file = os.path.join(dir_path, list_file)   
        f = open(list_file, "r")
        for _line in f:
            _rel_path = _line.split('\n')[0]
            self.test_set.add(_rel_path)
        f.close()

    def __call__(self, sample):
        if ("{}/{}.{}".format(sample.lbl, sample.name, sample.ext) 
            in self.test_set):
            return True
        else:
            return False

# set the same as test set
class for_val(object):
    def __init__(self, split="1"):
        self.for_test = for_test(split=split)
    def __call__(self, sample):
        return(self.for_test(sample))


if __name__ == "__main__":
    train_filter = for_train()
    print(train_filter.train_set)