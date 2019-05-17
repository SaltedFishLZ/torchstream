import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# ------------------------------------------------------------------------ #
#                   Labels and Corresponding CIDs                          #
# ------------------------------------------------------------------------ #

__labels__ = dict()
list_file = os.path.join(dir_path, "hmdb_labels.txt")
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
# From the paper we can know each class has at least 101 samples
# Page 2 of the paper
# paper link:
# http://serre-lab.clps.brown.edu/wp-content/uploads/2012/08/Kuehne_etal_iccv11.pdf
# So we estimate it as [101, intmax]
intmax = int(2**31)
__sample_num_per_class__ = dict(zip(__labels__.keys(), 51*[[101, intmax]]))


if __name__ == "__main__":
    print("All Labels & Corresponding CIDs")
    print(__labels__)
    print("Sample Number Per Class")
    print(__sample_num_per_class__)
