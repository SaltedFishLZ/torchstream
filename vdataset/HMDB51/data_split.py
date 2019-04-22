import os
import copy

from .label_map import label_map

__supported_splits__ = ["1", "2", "3"]


dir_path = os.path.dirname(os.path.realpath(__file__))
splits_folder = os.path.join(dir_path, "test_train_splits")

aux_dict = dict()
for _split in __supported_splits__:
    # create helping sets for each split
    train_set = set()
    test_set = set()
    for _label in list(label_map.keys()):
        _file = "{}_test_split{}.txt".format(_label, _split)
        _file = os.path.join(splits_folder, _file)
        f = open(_file, "r")
        for _line in f:
            text = _line.split('\n')[0]
            text = text.split(' ')
            _name = text[0]
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

def for_train(sample, split='1'):
    assert split in __supported_splits__, "Unsupported split"
    _name = sample.name
    _label = sample.lbl
    _rec = '_'.join([_label, _name])
    if (_rec in aux_dict[split]["train_set"]):
        return True
    else:
        return False

def for_test(sample, split='1'):
    assert split in __supported_splits__, "Unsupported split"
    _rec = {"name": sample.name, "label": sample.lbl}
    if (_rec in aux_dict[split]["test_set"]):
        return True
    else:
        return False

def for_val(sample, split='1'):
    return(for_test(sample, split))

