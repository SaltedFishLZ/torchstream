"""
Dataset Path
"""
__all__ = [
    "AVI_DATA_PATH", "JPG_DATA_PATH",
    "RAW_DATA_PATH", "PRC_DATA_PATH"
]


import os

USER_HOME = os.path.expanduser("~")

# Default setting
# NOTE
# If you have your seetings, please change it
AVI_DATA_PATH = os.path.join(USER_HOME, "Datasets", "HMDB51", "HMDB51-avi")
JPG_DATA_PATH = os.path.join(USER_HOME, "Datasets", "HMDB51", "HMDB51-jpg")
RAW_DATA_PATH = os.path.join(USER_HOME, "Datasets", "HMDB51", "HMDB51-raw")
PRC_DATA_PATH = os.path.join(USER_HOME, "Datasets", "HMDB51", "HMDB51-prc")
