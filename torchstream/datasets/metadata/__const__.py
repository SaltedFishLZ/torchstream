"""
Package Constants
"""

__all__ = [
    "ALLSET", "TRAINSET", "VALSET", "TESTSET",
    "IMGSEQ", "IMGEXT",
    "UNKNOWN_LABEL", "UNKNOWN_CID"
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

UNKNOWN_LABEL = "Unknown"
UNKNOWN_CID = -1
