"""Annotation data
"""
__all__ = [
    "__LABELS__", "__SAMPLES_PER_LABEL__",
    "__ANNOTATIONS__"
]

import os
import pickle
import logging

from . import __config__
from .csvparse import TRAINSET_DF, VALSET_DF, TESTSET_DF
from ...__const__ import UNKNOWN_LABEL, UNKNOWN_CID
from ....utils.filesys import touch_date

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# ---------------------------------------------------------------- #
#                  Configuring Python Logger                       #
# ---------------------------------------------------------------- #

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
if __config__.__VERY_VERY_VERBOSE__:
    logger.setLevel(logging.INFO)
elif __config__.__VERY_VERBOSE__:
    logger.setLevel(logging.WARNING)
elif __config__.__VERBOSE__:
    logger.setLevel(logging.ERROR)
else:
    logger.setLevel(logging.CRITICAL)

# ------------------------------------------------------------------------ #
#                   Labels and Corresponding CIDs                          #
# ------------------------------------------------------------------------ #

__SAMPLES_PER_LABEL__ = {
    "Doing other things"                : [1000, 12416],
    "Drumming Fingers"                  : [1000, 5444],
    "No gesture"                        : [1000, 5344],
    "Pulling Hand In"                   : [1000, 5379],
    "Pulling Two Fingers In"            : [1000, 5315],
    "Pushing Hand Away"                 : [1000, 5434],
    "Pushing Two Fingers Away"          : [1000, 5358],
    "Rolling Hand Backward"             : [1000, 5031],
    "Rolling Hand Forward"              : [1000, 5165],
    "Shaking Hand"                      : [1000, 5314],
    "Sliding Two Fingers Down"          : [1000, 5410],
    "Sliding Two Fingers Left"          : [1000, 5345],
    "Sliding Two Fingers Right"         : [1000, 5244],
    "Sliding Two Fingers Up"            : [1000, 5262],
    "Stop Sign"                         : [1000, 5413],
    "Swiping Down"                      : [1000, 5303],
    "Swiping Left"                      : [1000, 5160],
    "Swiping Right"                     : [1000, 5066],
    "Swiping Up"                        : [1000, 5240],
    "Thumb Down"                        : [1000, 5460],
    "Thumb Up"                          : [1000, 5457],
    "Turning Hand Clockwise"            : [1000, 3980],
    "Turning Hand Counterclockwise"     : [1000, 4181],
    "Zooming In With Full Hand"         : [1000, 5307],
    "Zooming In With Two Fingers"       : [1000, 5355],
    "Zooming Out With Full Hand"        : [1000, 5330],
    "Zooming Out With Two Fingers"      : [1000, 5379],
}

__LABELS__ = sorted(list(__SAMPLES_PER_LABEL__.keys()))

# generating label-cid mapping
# map "Doing other things" cid 0
CIDS = list(range(len(__LABELS__)))
CIDS = CIDS[1:len(CIDS)] + [0]
__LABELS__ = dict(zip(__LABELS__, CIDS))

# add UNKNOWN LABEL
__LABELS__[UNKNOWN_LABEL] = UNKNOWN_CID
__SAMPLES_PER_LABEL__[UNKNOWN_LABEL] = [0, 999999]



# ------------------------------------------------------------------------ #
#                 Collect Annotations for Each Sample                      #
# ------------------------------------------------------------------------ #

# NOTE: __annotations__ is a Python key word
# Currently, this dataset only provides annotation for training & validation
# We use None to mark unlabelled samples
__ANNOTATIONS__ = dict()

ANNOT_FILE = os.path.join(DIR_PATH, "jester-v1.annot")
if (os.path.exists(ANNOT_FILE)
        and (touch_date(FILE_PATH) < touch_date(ANNOT_FILE))
   ):
    logger.info("Find valid annotation cache")
    fin = open(ANNOT_FILE, "rb")
    __ANNOTATIONS__ = pickle.load(fin)
    fin.close()
else:
    logger.info("Building annotation data...")
    ## training/validation set has labels
    for df in (TRAINSET_DF, VALSET_DF):
        for idx, row in df.iterrows():
            video = str(row["video"])
            label = str(row["label"])
            __ANNOTATIONS__[video] = label
    ## testing set doesn't have labels
    for df in (TESTSET_DF, ):
        for idx, row in df.iterrows():
            video = str(row["video"])
            __ANNOTATIONS__[video] = UNKNOWN_LABEL
    ## TODO: write failure check
    fout = open(ANNOT_FILE, "wb")
    pickle.dump(__ANNOTATIONS__, fout)
    fout.close()

## Self Test Function
def test():
    """Self-testing function
    """
    print(len(__ANNOTATIONS__))
    print(__LABELS__)

if __name__ == "__main__":
    test()
