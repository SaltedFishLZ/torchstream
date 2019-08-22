"""JSON File Parsing Module
"""
import os
import json

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


fname = os.path.join(DIR_PATH, "something-something-v2-train.json")
TRAINSET_JLIST = []
with open(fname, "r", encoding="utf-8") as fin:
    TRAINSET_JLIST = json.load(fin)

fname = os.path.join(DIR_PATH, "something-something-v2-validation.json")
VALSET_JLIST = []
with open(fname, "r", encoding="utf-8") as fin:
    VALSET_JLIST = json.load(fin)

fname = os.path.join(DIR_PATH, "something-something-v2-test.json")
TESTSET_JLIST = []
with open(fname, "r", encoding="utf-8") as fin:
    TESTSET_JLIST = json.load(fin)


## Self Test Function
def test():
    """
    Self-test function
    """
    print("[Training Set List]")
    print(TRAINSET_JLIST)
    print("[Validation Set List]")
    print(VALSET_JLIST)
    print("[Testing Set List]")
    print(TESTSET_JLIST)
