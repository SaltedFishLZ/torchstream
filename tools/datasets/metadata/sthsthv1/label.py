import os
import pickle
import collections

from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download/")
DOWNLOAD_SRC_DIR = "tools/datasets/metadata/sthsthv1"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


# download metadata if there is no local cache
label_file = "something-something-v1-labels.csv"
label_path = os.path.join(DIR_PATH, label_file)
if not os.path.exists(label_path):
    label_src = os.path.join(
        DOWNLOAD_SERVER_PREFIX,
        DOWNLOAD_SRC_DIR,
        label_file
    )
    download(label_src, label_path)

# build class label to class id mapping (a dictionary)
labels = []
with open(label_path, "r") as fin:
    for _line in fin:
        text = _line.split('\n')[0]
        labels.append(text)
labels = sorted(labels)

class_to_idx = collections.OrderedDict()
for cid, label in enumerate(labels):
    class_to_idx[label] = cid


if __name__ == "__main__":
    print(class_to_idx)

    cls2idx_path = os.path.join(DIR_PATH, "sthsthv1_class_to_idx.pkl")    
    with open(cls2idx_path, "wb") as fout:
        pickle.dump(class_to_idx, fout)