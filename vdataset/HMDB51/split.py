import os
import copy

dir_path = os.path.dirname(os.path.realpath(__file__))

from .label import __labels__

# Split Option is a feature of HMDB51
# HMDB51 is not so large, so the authors give 3 split options for Train/Test
# dataset split.
# Different official split options are stored as 
# '<class name>_<test/train>_split?.txt' in splits_folder
__supported_split_options__ = ["1", "2", "3"]
split_options_folder = os.path.join(dir_path, "split_options")




# ---------------------------------------------------------------- #
#     Building Training/Test/Validation Sets for Split Options     #
# ---------------------------------------------------------------- #

# training/test sets of different split options
trainsets = []
testsets = []

for _split_option in __supported_split_options__:
    # create temporary sets for each split_option
    trainset = set()
    testset = set()

    for _label in list(__labels__.keys()):
        
        _file = "{}_test_split{}.txt".format(_label, _split_option)
        _file = os.path.join(split_options_folder, _file)
        
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

    trainsets.append(trainset)
    testsets.append(testset)





# ---------------------------------------------------------------- #
#               Main Classes (To Be Used Externally)               #        
# ---------------------------------------------------------------- #

class DatasetFilter(object):
    '''
    This is an abstract class of common codes for different splits
    '''
    def __init__(self, split="train", split_option="1"):
        assert split_option in __supported_split_options__, \
            "Unsupported split option"
        self.split = split
        self.split_option = split_option

    def __call__(self, sample):
        _name = sample.name
        _label = sample.lbl
        _rec = '_'.join([_label, _name])
        _split_option = int(self.split_option) - 1
        if ("train" == self.split):
            if (_rec in trainsets[_split_option]):
                return True
            else:
                return False
        elif ("test" == self.split):
            if (_rec in testsets[_split_option]):
                return True
            else:
                return False
        else:
            assert NotImplementedError

class TrainsetFilter(DatasetFilter):
    def __init__(self, split_option="1"):
        super(TrainsetFilter, self).__init__(split="train", split_option="1")

class TestsetFilter(DatasetFilter):
    def __init__(self, split_option="1"):
        super(TestsetFilter, self).__init__(split="test", split_option="1")

class ValsetFilter(DatasetFilter):
    def __init__(self, split_option="1"):
        # set the same as test set
        super(ValsetFilter, self).__init__(split="test", split_option="1")




if __name__ == "__main__":
    print("[Split 1]")
    print("Train - {}".format(len(trainsets[0])))
    print("Test - {}".format(len(testsets[0])))

    print("[Split 2]")
    print("Train - {}".format(len(trainsets[1])))
    print("Test - {}".format(len(testsets[1])))

    print("[Split 3]")
    print("Train - {}".format(len(trainsets[2])))
    print("Test - {}".format(len(testsets[2])))
