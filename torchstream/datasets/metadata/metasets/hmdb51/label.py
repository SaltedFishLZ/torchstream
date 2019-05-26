"""Annotation data
"""
__all__ = [
    "__LABELS__", "__SAMPLES_PER_LABEL__"
]

import os

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

# ------------------------------------------------------------------------ #
#                   Labels and Corresponding CIDs                          #
# ------------------------------------------------------------------------ #

__LABELS__ = dict()
LIST_FILE = os.path.join(DIR_PATH, "hmdb_labels.txt")
f = open(LIST_FILE, "r")
for _line in f:
    text = _line.split('\n')[0]
    text = text.split(' ')
    __LABELS__[text[1]] = int(text[0]) - 1
f.close()


# ------------------------------------------------------------------------ #
#           Sample Number Per Class (Useful for Integrity Check)           #
# ------------------------------------------------------------------------ #
#
# From the paper we can know each class has at least 101 samples
# Page 2 of the paper
# paper link:
# http://serre-lab.clps.brown.edu/wp-content/uploads/2012/08/Kuehne_etal_iccv11.pdf
# So we estimate it as [101, INT_MAX]
INT_MAX = int(2**31)
__SAMPLES_PER_LABEL__ = dict(zip(__LABELS__.keys(), 51*[[101, INT_MAX]]))


if __name__ == "__main__":
    print("All Labels & Corresponding CIDs")
    print(__LABELS__)
    print("Sample Number Per Class")
    print(__SAMPLES_PER_LABEL__)
