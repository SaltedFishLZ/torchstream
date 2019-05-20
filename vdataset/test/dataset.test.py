"""
"""
import os
import sys

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)
PKG_PATH = os.path.dirname(DIR_PATH)

sys.path.append(os.path.dirname(PKG_PATH))
from vdataset import dataset

dataset.test()
