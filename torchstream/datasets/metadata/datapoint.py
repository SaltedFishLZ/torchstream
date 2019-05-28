"""Abstract Datapoint Class
"""
import os
import copy
import logging
import collections

from . import __config__
from .__const__ import UNKNOWN_LABEL
from .__support__ import __SUPPORTED_MODALITIES__, \
    __SUPPORTED_IMAGES__, __SUPPORTED_VIDEOS__
from ..utils.filesys import strip_extension


# ---------------------------------------------------------------- #
#                  Configuring Python Logger                       #
# ---------------------------------------------------------------- #

if __config__.__VERY_VERBOSE__:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s"
    )
elif __config__.__VERY_VERBOSE__:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s - %(levelname)s - %(message)s"
    )
elif __config__.__VERBOSE__:
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
else:
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
logger = logging.getLogger(__name__)


class DataPoint(object):
    """
    An video sample struct containing the meta-data of a video sample
    """
    #  @param root str: absolute root path of the dataset
    #  @param path str: absolute path of 1 video sample
    #  @param name str: file name (without any extension and path)
    #  @param ext str:  file extension (e.g., "avi", "mp4"), '.' excluded.
    #  @param label str:  
    #      label of the sample, is a unique string in certain dataset
    #  @param cid str:  
    #      class id of the sample, is the numerical representation of label
    def __init__(self, root, rpath, name, label=UNKNOWN_LABEL,
                 mod="RGB", ext="jpg"):
        assert isinstance(root, str), TypeError
        assert isinstance(rpath, str), TypeError
        assert isinstance(name, str), TypeError
        assert isinstance(mod, str), TypeError
        assert isinstance(ext, str), TypeError
        assert isinstance(label, str), TypeError

        if __config__.__STRICT__:
            assert mod in __SUPPORTED_MODALITIES__, NotImplementedError
            assert ext in __SUPPORTED_MODALITIES__[mod], NotImplementedError
        self.root = root
        self.rpath = rpath
        self.name = name
        self.mod = mod
        self.ext = ext
        self.label = label
        self.seq = ext in __SUPPORTED_IMAGES__[mod]
        self.path = os.path.join(root, rpath)

    def __repr__(self, idents=0):
        string = idents * "\t" + "DataPoint: \n"
        string += idents * "\t" + str(self.name)
        if self.seq:
            string += "(frame sequence)"
        string += '\n'
        string += idents * "\t" + "[label] : {}  \t".format(self.label)
        string += idents * "\t" + "[path] : {}".format(self.path)
        return string
    
    def __eq__(self, other):
        if isinstance(other, DataPoint):
            return(
                (self.root == other.root)
                and (self.path == other.path)
                and (self.name == other.name)
                and (self.mod == other.mod)
                and (self.ext == other.ext)
                and (self.seq == other.seq)
                and (self.label == other.label)
            )
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        assert isinstance(other, DataPoint), TypeError
        
        try:
            name_0 = int(self.name)
        except ValueError:
            name_0 = self.name
        try:
            name_1 = int(other.name)
        except ValueError:
            name_1 = other.name
        
        ## both name can be converted to int
        if isinstance(name_0, int) and isinstance(name_1, int):
            return name_0 < name_1
        ## compare order: label -> name
        else:
            if self.label != other.label:
                return self.label < other.label
            else:
                return self.name < other.name

    def root_migrated(self, new_root):
        """
        Get a new DataPoint object with dataset root migrated to a new one.
        """
        return DataPoint(root=new_root, rpath=self.rpath,
                         name=self.name, mod=self.mod,
                         ext=self.ext, label=self.label
                         )


    def extension_migrated(self, ext):
        """
        Get a new DataPoint object with file extension migrated
        modality must be the same.
        """
        assert ext in __SUPPORTED_MODALITIES__[self.mod], NotImplementedError
        
        def _to_seq(sample, ext):
            """ any sample -> sequence sample
            """
            sample.seq = True
            sample.ext = ext
            sample.rpath = strip_extension(sample.rpath)
            sample.path = strip_extension(sample.path)
            return sample
        
        def _to_vid(sample, ext):
            """any sample -> video sample
            """
            sample.seq = False
            sample.ext = ext
            sample.rpath = strip_extension(sample.rpath) + "." + ext
            sample.path = strip_extension(sample.path) + "." + ext
            return sample

        datapoint = copy.deepcopy(self)
        if ext in __SUPPORTED_IMAGES__[self.mod]:
            return _to_seq(datapoint, ext)
        elif ext in __SUPPORTED_VIDEOS__[self.mod]:
            return _to_vid(datapoint, ext)
        else:
            raise NotImplementedError


class DataPointCounter(collections.Counter):
    """A set of data pionts with some statistics information

    Args:
        datapoints (list): a list of DataPoint objects
    """
    def __init__(self, datapoints):
        labels = []
        for datapoint in datapoints:
            labels.append(datapoint.label)

        super(DataPointCounter, self).__init__(labels)
        self.datapoints = datapoints

    def __repr__(self):
        return collections.Counter.__repr__(self)

    ## Filter samples using a filter.
    def filter_samples(self, filter_):
        """Remove samples according to a given filter

        Args:
            filter_ (function): A filter function taking 1 DataPoint object as
                input and return whether the sample should be preserved (True
                or False)
        """
        # REMOVED
        # TODO
        raise NotImplementedError

    def update_labels(self, labels):
        """Update labels and corresponding counts. 
        """
        # REMOVED
        # TODO
        raise NotImplementedError

    def update_samples(self, samples):
        """Update samples
        """
        # REMOVED
        # TODO
        raise NotImplementedError

    ## Documentation for a method.
    #  @param self The object pointer.
    def refresh_statistics(self):
        """
        Update self.counts to make all statstistics consistent when you change
        the data without using member functions
        """
        ## REMVOED
        ## TODO




# ------------------------------------------------------------------------- #
#              Self-test Utilities (Not To Be Used outside)                 #
# ------------------------------------------------------------------------- #

def test_sample():
    a = DataPoint(root="Foo", rpath="Bar", name="test", ext="avi")
    print(a)
    b = a.root_migrated("Fooood")
    print(b)
    c = b.extension_migrated(ext="jpg")
    print(c)
    d = DataPoint(root="Foo", rpath="Bar", name="aha", ext="avi")
    d1000 = DataPoint(root="Foo", rpath="Bar", name="1000", ext="avi")
    d9 = DataPoint(root="Foo", rpath="Bar", name="9", ext="avi")
    print("Test Equality", a == c)
    print("Test Order", a < c, c < a, d < a, d9 < d1000)




if __name__ == "__main__":

    test_sample()
