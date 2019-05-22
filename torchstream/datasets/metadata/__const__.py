"""
Package Constants
"""

__all__ = [
    "ALLSET", "TRAINSET", "VALSET", "TESTSET",
    "IMGSEQ", "IMGEXT",
    "UNKOWN_LABEL", "UNKOWN_CID"
]


# ---------------- #
#  Dataset Splits  #
# ---------------- #

ALLSET = 0
TRAINSET = 1
VALSET = 2
TESTSET = 3


# ---------------- #
#     Meta-data    #
# ---------------- #

IMGSEQ = "jpg"
IMGEXT = "jpg"

UNKOWN_LABEL = None
UNKOWN_CID = -1
