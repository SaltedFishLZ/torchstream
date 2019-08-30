"""Split Settings for HMDB51
HMDB51 has 3 split plans for training & test splitting.
NOTE: Read split_readme.txt for more details.
This is the HMDB official annotation:
http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/split_readme.txt
"""
import os
import glob

from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = "a18:/home/eecs/zhen/video-acc/download/"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

# download split annotations
split_annot_dir = "testTrainMulti_7030_splits"
split_dir_path = os.path.join(DIR_PATH, split_annot_dir)
if not os.path.exists(split_dir_path):
    split_dir_src = DOWNLOAD_SERVER_PREFIX + \
        "tools/datasets/metadata/hmdb51/{}".format(split_annot_dir)
    download(split_dir_src, split_dir_path)


def test_sample_names(split_num):
    """
    Args:
        split_num (int): split plan num
    Return:
        list of strings: a list of sample names of all testing samples of
        then given split plan.
    """
    assert isinstance(split_num, int), TypeError
    assert split_num in (1, 2, 3), ValueError("Invalid split numer")

    glob_str = os.path.join(split_dir_path,
                            "*_test_split{}.txt".format(split_num))
    annot_file_paths = glob.glob(glob_str)

    results = []
    for fpath in annot_file_paths:
        with open(fpath, "r") as fin:
            for _line in fin:
                _text = _line.split("\n")[0]    # remove \n
                _text = _text.split(" ")        # split name and type
                _name = _text[0]
                _name = _name.split('.')[0]     # remove file extension .avi
                _type = _text[1]
                # HMDB51's rule:
                # 2 = test set
                if (_type == "2"):
                    results.append(_name)
    return results


def train_sample_names(split_num):
    """
    Args:
        split_num (int): split plan num
    Return:
        list of strings: a list of sample names of all training samples of
        then given split plan.
    """
    assert isinstance(split_num, int), TypeError
    assert split_num in (1, 2, 3), ValueError("Invalid split numer")

    glob_str = os.path.join(split_dir_path,
                            "*_test_split{}.txt".format(split_num))
    annot_file_paths = glob.glob(glob_str)

    results = []
    for fpath in annot_file_paths:
        with open(fpath, "r") as fin:
            for _line in fin:
                _text = _line.split("\n")[0]    # remove \n
                _text = _text.split(" ")        # split name and type
                _name = _text[0]
                _name = _name.split('.')[0]     # remove file extension .avi
                _type = _text[1]
                # HMDB51's rule:
                # 1 = training set
                if (_type == "1"):
                    results.append(_name)
    return results


if __name__ == "__main__":
    sample_names = test_sample_names(split_num=1)
    print(len(sample_names))

    sample_names = train_sample_names(split_num=1)
    print(len(sample_names))
