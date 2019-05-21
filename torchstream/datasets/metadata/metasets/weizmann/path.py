"""
Paths to dataset store points
NOTE: It should not apear in release version
"""
__all__ = [
    "AVI_DATA_PATH", "JPG_DATA_PATH",
    "RAW_DATA_PATH", "PRC_DATA_PATH"
]

import os

USER_HOME = os.path.expanduser("~")

# Default setting
# NOTE
# If you have your settings, please change it
AVI_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Weizmann", "Weizmann-avi")
JPG_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Weizmann", "Weizmann-jpg")
RAW_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Weizmann", "Weizmann-raw")
PRC_DATA_PATH = os.path.join(USER_HOME, "Datasets", "Weizmann", "Weizmann-prc")
