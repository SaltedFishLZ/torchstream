"""
Dataset Path
"""
__all__ = [
    "AVI_DATA_PATH", "JPG_DATA_PATH",
    "RAW_DATA_PATH", "PRC_DATA_PATH"
]

import os

HOME = os.path.expanduser("~")

# Default setting
AVI_DATA_PATH = os.path.join(HOME, "Datasets", "UCF101", "UCF101-avi")
JPG_DATA_PATH = os.path.join(HOME, "Datasets", "UCF101", "UCF101-jpg")
RAW_DATA_PATH = os.path.join(HOME, "Datasets", "UCF101", "UCF101-raw")
PRC_DATA_PATH = os.path.join(HOME, "Datasets", "UCF101", "UCF101-prc")
