import os
import json

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

fname = os.path.join(DIR_PATH, "something-something-v2-train.json")
TRAINSET_JLIST = []
with open(fname, "r") as fin:
    TRAINSET_JLIST = json.load(fin)

fname = os.path.join(DIR_PATH, "something-something-v2-validation.json")
VALSET_JLIST = []
with open(fname, "r") as fin:
    VALSET_JLIST = json.load(fin)

fname = os.path.join(DIR_PATH, "something-something-v2-test.json")
TESTSET_JLIST = []
with open(fname, "r") as fin:
    TESTSET_JLIST = json.load(fin)


## Self Test Function
#  
#  Details
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
