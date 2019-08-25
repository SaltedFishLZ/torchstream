"""
Dataset Path
"""
__all__ = [
    "JPG_DATA_PATH", "AVI_DATA_PATH",
    "RAW_DATA_PATH", "PRC_DATA_PATH",
    "JPG_FILE_TMPL", "JPG_IDX_OFFSET"
]

import os

USER_HOME = os.path.expanduser("~")
LOCAL_DATASETS = os.path.join(USER_HOME, "Datasets-local")

# Default setting
JPG_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Jester", "Jester-v1-jpg")
AVI_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Jester", "Jester-v1-avi")
RAW_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Jester", "Jester-v1-raw")
PRC_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Jester", "Jester-v1-prc")


if os.path.exists(LOCAL_DATASETS):
    local_path = os.path.join(LOCAL_DATASETS, "Jester", "Jester-v1-jpg")
    if os.path.exists(local_path):
        JPG_DATA_PATH = local_path
    
if os.path.exists(LOCAL_DATASETS):
    local_path = os.path.join(LOCAL_DATASETS, "Jester", "Jester-v1-avi")
    if os.path.exists(local_path):
        AVI_DATA_PATH = local_path

# jpg file name template of the official datasets
JPG_FILE_TMPL = "{0:05d}"

# jpg frame index offset
JPG_IDX_OFFSET = 1