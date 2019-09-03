import os
import pickle
import collections

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

# download metadata if there is no local cache
label_file = "classInd.txt"
label_path = os.path.join(metadata_dir_path, label_file)
if not os.path.exists(label_path):
    label_src = os.path.join(
        DOWNLOAD_SERVER_PREFIX,
        DOWNLOAD_SRC_DIR,
        metadata_dir,
        label_file
    )
    download(label_src, label_path)

# build class label to class id mapping (a dictionary)
class_to_idx = collections.OrderedDict()
with open(label_path, "r") as fin:
    for _line in fin:
        text = _line.split('\n')[0]
        text = text.split(' ')
        class_to_idx[text[1]] = int(text[0]) - 1


if __name__ == "__main__":
    print(class_to_idx)

    cls2idx_path = os.path.join(DIR_PATH, "ucf101_class_to_idx.pkl")    
    with open(cls2idx_path, "wb") as fout:
        pickle.dump(class_to_idx, fout)
