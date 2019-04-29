import os
import copy

from .label import __labels__

__supported_splits__ = ["1", "2", "3"]


dir_path = os.path.dirname(os.path.realpath(__file__))
splits_folder = os.path.join(dir_path, "test_train_splits")

aux_dict = dict()
for _split in __supported_splits__:
    # create helping sets for each split
    train_set = set()
    test_set = set()
    for _label in list(__labels__.keys()):
        _file = "{}_test_split{}.txt".format(_label, _split)
        _file = os.path.join(splits_folder, _file)
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
                train_set.add(_rec)
            elif (text[1] == '2'):
                test_set.add(_rec)
        f.close()
    # for each split create a dictionary
    aux_dict[_split] = {
        'train_set': copy.deepcopy(train_set),
        'test_set': copy.deepcopy(test_set)}

class TrainsetFilter(object):
    def __init__(self, split='1'):
        assert split in __supported_splits__, "Unsupported split"
        self.split = split
    def __call__(self, sample):
        _name = sample.name
        _label = sample.lbl
        _rec = '_'.join([_label, _name])
        if (_rec in aux_dict[self.split]["train_set"]):
            return True
        else:
            return False

class TestsetFilter(object):
    def __init__(self, split='1'):
        assert split in __supported_splits__, "Unsupported split"
        self.split = split
    def __call__(self, sample):
        _name = sample.name
        _label = sample.lbl
        _rec = '_'.join([_label, _name])
        if (_rec in aux_dict[self.split]["test_set"]):
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
    print("Split 1:")
    print("Train - {}".format(len(aux_dict['1']['train_set'])))
    print("Test - {}".format(len(aux_dict['1']['test_set'])))

    print("Split 2:")
    print("Train - {}".format(len(aux_dict['2']['train_set'])))
    print("Test - {}".format(len(aux_dict['2']['test_set'])))
    