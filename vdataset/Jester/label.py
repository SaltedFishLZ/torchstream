"""
"""
import os
import pickle

from .. import constant
from .common import trainset_df, valset_df, testset_df
from ..utilities import modification_date, creation_date

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# ------------------------------------------------------------------------ #
#                   Labels and Corresponding CIDs                          #
# ------------------------------------------------------------------------ #

__classes__ = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Down",
    "Swiping Up",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Pushing Two Fingers Away",
    "Pulling Two Fingers In",
    "Rolling Hand Forward",
    "Rolling Hand Backward",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Thumb Up",
    "Thumb Down",
    "Shaking Hand",
    "Stop Sign",
    "Drumming Fingers",
    "No gesture",
    "Doing other things",
]

# generating label-cid mapping
# map "Doing other things" cid 0
cids = list(range(len(__classes__)))
cids = cids[1:len(cids)] + [0]
__labels__ = dict(zip(__classes__, cids))



__sample_num_per_class__ = {
    "Doing other things"                : [1000, 12416]  ,
    "Drumming Fingers"                  : [1000, 5444 ]  ,
    "No gesture"                        : [1000, 5344 ]  ,
    "Pulling Hand In"                   : [1000, 5379 ]  ,
    "Pulling Two Fingers In"            : [1000, 5315 ]  ,
    "Pushing Hand Away"                 : [1000, 5434 ]  ,
    "Pushing Two Fingers Away"          : [1000, 5358 ]  ,
    "Rolling Hand Backward"             : [1000, 5031 ]  ,
    "Rolling Hand Forward"              : [1000, 5165 ]  ,
    "Shaking Hand"                      : [1000, 5314 ]  ,
    "Sliding Two Fingers Down"          : [1000, 5410 ]  ,
    "Sliding Two Fingers Left"          : [1000, 5345 ]  ,
    "Sliding Two Fingers Right"         : [1000, 5244 ]  ,
    "Sliding Two Fingers Up"            : [1000, 5262 ]  ,
    "Stop Sign"                         : [1000, 5413 ]  ,
    "Swiping Down"                      : [1000, 5303 ]  ,
    "Swiping Left"                      : [1000, 5160 ]  ,
    "Swiping Right"                     : [1000, 5066 ]  ,
    "Swiping Up"                        : [1000, 5240 ]  ,
    "Thumb Down"                        : [1000, 5460 ]  ,
    "Thumb Up"                          : [1000, 5457 ]  ,
    "Turning Hand Clockwise"            : [1000, 3980 ]  ,
    "Turning Hand Counterclockwise"     : [1000, 4181 ]  ,
    "Zooming In With Full Hand"         : [1000, 5307 ]  ,
    "Zooming In With Two Fingers"       : [1000, 5355 ]  ,
    "Zooming Out With Full Hand"        : [1000, 5330 ]  ,
    "Zooming Out With Two Fingers"      : [1000, 5379 ]  ,
}



# ------------------------------------------------------------------------ #
#                 Collect Annotations for Each Sample                      #
# ------------------------------------------------------------------------ #

# NOTE:
# __annotations__ is a Python key word
# So, we use __targets__
# Currently, this dataset only provides annotation for training & validation
# We use None to mark unlabelled samples
__targets__ = dict()

annot_file = os.path.join(DIR_PATH, "jester-v1.annot")
if (os.path.exists(annot_file)
    and (creation_date(FILE_PATH) < creation_date(annot_file))
    and (modification_date(FILE_PATH) < modification_date(annot_file))):
    print("Find valid annotation cache")
    f = open(annot_file, "rb")
    __targets__ = pickle.load(f)
    f.close()
else:
    for df in (trainset_df, valset_df):
        for idx, row in df.iterrows():
            video = str(row["video"])
            label = str(row["label"])
            __targets__[video] = label
    for df in (testset_df, ):
        for idx, row in df.iterrows():
            video = str(row["video"])
            __targets__[video] = constant.LABEL_UNKOWN
    # TODO: consistency issue    
    f = open(annot_file, "wb")
    pickle.dump(__targets__, f)
    f.close()



if __name__ == "__main__":    
    print(len(__targets__))
    print(cids)