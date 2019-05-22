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
LIST_FILE = os.path.join(DIR_PATH, "classInd.txt")
fin = open(LIST_FILE, "r")
for _line in fin:
    text = _line.split('\n')[0]
    text = text.split(' ')
    __LABELS__[text[1]] = int(text[0])
fin.close()


# ------------------------------------------------------------------------ #
#           Sample Number Per Class (Useful for Integrity Check)           #
# ------------------------------------------------------------------------ #
#
# Details from UCF's official website
# https://www.crcv.ucf.edu/data/UCF101.php
# "The videos in 101 action categories are grouped into 25 groups, where 
# each group can consist of 4-7 videos of an action. The videos from the
# same group may share some common features, such as similar background,
# similar viewpoint, etc."
# So, we choose to use [25 * 4, 25 * 7] as the sample num's [min, max] for
# each class
__SAMPLES_PER_LABEL__ = dict(zip(__LABELS__.keys(), 101*[[25*4, 25*7]]))



if __name__ == "__main__":
    print("All Labels & Corresponding CIDs")
    print(__LABELS__)
    print("Sample Number Per Class")
    print(__SAMPLES_PER_LABEL__)
