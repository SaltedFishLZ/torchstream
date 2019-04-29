import os

# ------------------------------------------------------------------------ #
#                   Labels and Corresponding CIDs                          #
# ------------------------------------------------------------------------ #

__labels__ = dict()
dir_path = os.path.dirname(os.path.realpath(__file__))
list_file = os.path.join(dir_path, "classInd.txt")
f = open(list_file, "r")
for _line in f:
    text = _line.split('\n')[0]
    text = text.split(' ')
    __labels__[text[1]] = int(text[0])
f.close()


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
__sample_num_per_class__ = dict(zip(__labels__.keys(), 101*[[25*4, 25*7]]))



if __name__ == "__main__":
    print("All Labels & Corresponding CIDs")
    print(__labels__)
    print("Sample Number Per Class")
    print(__sample_num_per_class__)