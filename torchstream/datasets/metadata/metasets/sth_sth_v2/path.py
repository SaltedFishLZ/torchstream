"""
Dataset Path
"""
__all__ = [
    "WEBM_DATA_PATH", "AVI_DATA_PATH",
    "JPG_DATA_PATH",
    "RAW_DATA_PATH", "PRC_DATA_PATH"
]

import os

USER_HOME = os.path.expanduser("~")

# Default setting
# NOTE
# If you have your seetings, please change it
WEBM_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Sth-sth", "Sth-sth-v2-webm")
AVI_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Sth-sth", "Sth-sth-v2-avi")
JPG_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Sth-sth", "Sth-sth-v2-jpg")
RAW_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Sth-sth", "Sth-sth-v2-raw")
PRC_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Sth-sth", "Sth-sth-v2-avi")