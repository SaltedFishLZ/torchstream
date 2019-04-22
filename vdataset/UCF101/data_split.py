import os

__supported_splits__ = ["1", "2", "3"]

dir_path = os.path.dirname(os.path.realpath(__file__))

def for_train(sample, split="1"):
    assert (split in __supported_splits__), "Unsupported split"
    list_file = "trainlist0" + split + ".txt"
    list_file = os.path.join(dir_path, list_file)
    
    train_set = set()
    f = open(list, "r")
    for _line in f:
        _rel_path = _line.split('\n')[0]
        train_set.add(_rel_path)
    f.close()

    if ("{}/{}.{}".format(sample.lbl, sample.name, sample.ext) in train_set):
        return True
    else:
        return False

def for_test(sample, split="1"):
    assert (split in __supported_splits__), "Unsupported split"
    list_file = "testlist0" + split + ".txt"
    list_file = os.path.join(dir_path, list_file)
    
    test_set = set()
    f = open(list, "r")
    for _line in f:
        _rel_path = _line.split('\n')[0]
        test_set.add(_rel_path)
    f.close()

    if ("{}/{}.{}".format(sample.lbl, sample.name, sample.ext) in test_set):
        return True
    else:
        return False

# set the same as test set
def for_val(sample, split="1"):
    return(for_train(sample, split))