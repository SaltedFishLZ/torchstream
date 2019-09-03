import os

from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download/")
DOWNLOAD_SRC_DIR = "tools/datasets/metadata/ucf101"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

# create metadata directory if not exists
metadata_dir = "ucfTrainTestlist"
metadata_dir_path = os.path.join(DIR_PATH, metadata_dir)
if not os.path.exists(metadata_dir_path):
    os.makedirs(metadata_dir_path, exist_ok=True)
else:
    assert os.path.isdir(metadata_dir_path)


def get_sample_names(split, split_num):
    """
    Args:
        split (str): split ("train" or "test")
        split_num (int): split plan num
    Return:
        list of strings: a list of sample names of all samples of
        then given split plan.
    """
    assert isinstance(split, str), TypeError
    assert split in ["train", "test"], ValueError("Invalid split name")
    assert isinstance(split_num, int), TypeError
    assert split_num in (1, 2, 3), ValueError("Invalid split numer")

    split_file = "{}list{:02d}.txt".format(split, split_num)
    split_path = os.path.join(metadata_dir_path, split_file)

    # download metadata if there is no local cache
    if not os.path.exists(split_path):
        split_src = os.path.join(
            DOWNLOAD_SERVER_PREFIX,
            DOWNLOAD_SRC_DIR,
            metadata_dir,
            split_file
        )
        download(split_src, split_path)

    results = []
    with open(split_path, "r") as fin:
        for _line in fin:
            _text = _line.split('\n')[0]    # remove \n
            _path = _text.split(' ')[0]     # remove class id
            _file = _path.split('/')[1]     # remove directory
            _name = _file.split('.')[0]     # remove file extension
            results.append(_name)

    return results


if __name__ == "__main__":
    sample_names = get_sample_names(split="train", split_num=1)
    print(len(sample_names))

    sample_names = get_sample_names(split="test", split_num=1)
    print(len(sample_names))
