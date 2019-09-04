import os

from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download/")
DOWNLOAD_SRC_DIR = "tools/datasets/metadata/sthsthv1"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

# -------------------------------- #
# download annotations if there is #
# no local cache.                  #
# -------------------------------- #

# training set annotations
train_annot_file = "something-something-v1-train.csv"
train_annot_path = os.path.join(DIR_PATH, train_annot_file)
if not os.path.exists(train_annot_path):
    train_annot_src = os.path.join(
        DOWNLOAD_SERVER_PREFIX,
        DOWNLOAD_SRC_DIR,
        train_annot_file
    )
    download(train_annot_src, train_annot_path)

# validation set annotations
val_annot_file = "something-something-v1-validation.csv"
val_annot_path = os.path.join(DIR_PATH, val_annot_file)
if not os.path.exists(val_annot_path):
    val_annot_src = os.path.join(
        DOWNLOAD_SERVER_PREFIX,
        DOWNLOAD_SRC_DIR,
        val_annot_file
    )
    download(val_annot_src, val_annot_path)

# -------------------------------- #
#   build annotation dictionaries  #
# -------------------------------- #

# construct train annotations
train_annot_dict = dict()
with open(train_annot_path, "r") as fin:
    for _line in fin:
        _text = _line.split('\n')[0]
        _name, _label = _text.split(';')
        train_annot_dict[_name] = _label

# construct val annotations
val_annot_dict = dict()
with open(val_annot_path, "r") as fin:
    for _line in fin:
        _text = _line.split('\n')[0]
        _name, _label = _text.split(';')
        val_annot_dict[_name] = _label

# merge into full annotations
full_annot_dict = dict()
for _k in train_annot_dict:
    full_annot_dict[_k] = train_annot_dict[_k]
for _k in val_annot_dict:
    full_annot_dict[_k] = val_annot_dict[_k]
