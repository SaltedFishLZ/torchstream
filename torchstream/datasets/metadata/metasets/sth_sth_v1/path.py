"""
Dataset Path
"""
__all__ = [
    "JPG_DATA_PATH",
    "RAW_DATA_PATH", "PRC_DATA_PATH"
]

import os

HOME = os.path.expanduser("~")

# Default setting
JPG_DATA_PATH = os.path.join(HOME, "Datasets", "Sth-sth", "Sth-sth-v1-jpg")
RAW_DATA_PATH = os.path.join(HOME, "Datasets", "Sth-sth", "Sth-sth-v1-raw")
PRC_DATA_PATH = os.path.join(HOME, "Datasets", "Sth-sth", "Sth-sth-v1-prc")
