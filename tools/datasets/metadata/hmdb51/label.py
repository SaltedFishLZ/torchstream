import os
import collections

from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = "a18:/home/eecs/zhen/video-acc/download/"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

# download labels
label_file = "hmdb51_labels.txt"
label_path = os.path.join(DIR_PATH, label_file)
if not os.path.exists(label_path):
    label_src = DOWNLOAD_SERVER_PREFIX + \
        "tools/datasets/metadata/hmdb51/{}".format(label_file)
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
