# -*- coding: utf-8 -*-

ALLSET = 0
TRAINSET = 1
VALSET = 2
TESTSET = 3

IMGSEQ = ""
IMGEXT = "jpg"

LABEL_UNKOWN= None
CID_UNKOWN = -1

# ----------------------------------------------------------------- #
#           Runtime Settings for Python Scripts                     #
# ----------------------------------------------------------------- #
# NOTE: __debug__ parameter cannot be overwritten in normal Python 
# scrips, so we don't modify it here. You should specify it via -O
# parameter when you run Python intepreter.
# * in __test__ mode, all scripts will perform self-test
# * in __profile__ mode, all scripts will counting execution time and 
#   report time break down
# * in __strict__ mode, all scripts will have more strict santity check 
#   to make sure you use it as the intended way
# * in __verbose__ mode, all scripts will use Python logging module to 
#   log some critical information. while you may specify whether the 
#   logging module needs to dump the log info or not.
# * in __vverbose__ (very verbose) mode, all scripts will print detailed
#   information (including but not limited to __verbose__ information)
#   in stdout.
__test__ = True
__profile__ = True
__strict__ = True
__verbose__ = False
__vverbose__ = False

# ----------------------------------------------------------------- #
#           Input Data Modality & File Name Extension               #
# ----------------------------------------------------------------- #
# supported input data modality and corresponding file extensions
__supported_modalities__ = ['RGB']
__supported_modality_files__ = {
    'RGB': [IMGSEQ, "jpg", "avi", "mp4"]
    }
__supported_video_files__ = {
    'RGB' : ['avi', 'mp4']
}
# here, "GRAY" means single-channel data
# some optical-flow based methods may store flow files in jpg
__supported_color_space__ = ["BGR", "RGB", "GRAY"]

# ----------------------------------------------------------------- #
#           Dataset Structure Style and Supporting                  #
# ----------------------------------------------------------------- #
# NOTE: You shall not store other files in the dataset !!! 
#
# * 1. UCF101 style:
#   Your video dataset must have the following file orgnization:
#   Data Root
#   ├── Class 0
#   │   ├── Video 0
#   |   ├── Video 1
#   |   ├── ...
#   |   └── Video N_0
#   ...
#   |
#   └── Class K ...
#   If you use split image frames rather than an entire video, 
#   {Video i} shall be a folder contain all frames in order.
#   for example:
#   ├── Class 0
#   │   ├── Video 0
#   |   |   ├── 0.jpg
#   |   |   ├── 1.jpg
#   ...
#   Or you can storage video files like video_0.mp4
#   These should be specified via [use_imgs]
#   This style applies to the following datasets:
#   * UCF101
#   * HMDB51
#   * Weizmann
#   
# * 2. Kinetics style:
#   Kinetics Dataset already split training, validation, testing into
#   different folders ["train", "val", "test"]. Currently, the test set
#   has no annotations. While training set and validation set each follows
#   the UCF101 style
#
# * 3. 20BN style
#   Datasets from 20BN (a company) like Sth-sth and Jester usually have the
#   following structure:
#   Data Root
#   ├── Video 0
#   ├── Video 1
#   ├── ...
#   └── Video N
#   And there are additional annotation data for each video.
#   It applies to the following datasets:
#   * Something-something V1 & V2
#   * Jester
__supported_dataset_styles__ = ['UCF101', '20BN']

# key: dataset name, value: structure styles
__supported_datasets__ = {
    # UCF101 styled datasets
    'UCF101':'UCF101', 'HMDB51':'UCF101', 'Weizmann':'UCF101',
    'Jester':'20BN', 'Sth-sth-v1':'20BN', 'Sth-sth-v2':'20BN',
}
