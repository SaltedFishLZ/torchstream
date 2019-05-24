"""
Dataset Path
"""
__all__ = [
    "JPG_DATA_PATH",
    "RAW_DATA_PATH", "PRC_DATA_PATH",
    "JPG_FILE_TMPL", "JPG_IDX_OFFSET"
]

import os

USER_HOME = os.path.expanduser("~")

# Default setting
JPG_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Jester", "Jester-v1-jpg")
RAW_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Jester", "Jester-v1-raw")
PRC_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Jester", "Jester-v1-prc")

# jpg file name template of the official datasets
JPG_FILE_TMPL = "{0:05d}"

# jpg frame index offset
JPG_IDX_OFFSET = 1