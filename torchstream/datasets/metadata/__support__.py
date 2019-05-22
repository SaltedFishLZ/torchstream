"""
Metadata support information
"""
__all__ = [
    "__SUPPORTED_MODALITIES__",
    "__SUPPORTED_VIDEOS__", "__SUPPORTED_IMAGES__",
]


# ----------------------------------------------------------------- #
#           Input Data Modality & File Name Extension               #
# ----------------------------------------------------------------- #
# supported input data modality and corresponding file extensions
__SUPPORTED_VIDEOS__ = {
    "RGB" : ["avi", "mp4", "webm"]
}
__SUPPORTED_IMAGES__ = {
    "RGB" : ["jpg"]
}
__SUPPORTED_MODALITIES__ = {
    "RGB" : __SUPPORTED_VIDEOS__["RGB"] + __SUPPORTED_IMAGES__["RGB"]
}


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
    'ucf101':'UCF101', 'hmdb51':'UCF101', 'weizmann':'UCF101',
    # 20BN styled datasets 
    'jester_v1':'20BN', 'sth_sth_v1':'20BN', 
}