"""Abstract Datapoint Class
"""
import os
import copy
import logging
import collections

from . import __config__
from .__const__ import UNKNOWN_LABEL
from .__support__ import __SUPPORTED_IMAGES__, __SUPPORTED_VIDEOS__
from ..utils.filesys import strip_extension

# configuring logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(__config__.LOGGER_LEVEL)


class DataPoint(object):
    """Meta-data of a video sample in a certain dataset
    Args:
        root (str): root path of the dataset
        reldir (str): relative directory path of the sample in the dataset
        name (str): file name of the sample (without any extension and path)
        ext (str): file extension of the sample (e.g., "avi", "mp4").
            NOTE: '.' excluded.
        label (str): class label of this sample
    """
    def __init__(self, root, reldir, name, ext="jpg", label=UNKNOWN_LABEL):
        assert isinstance(root, str), TypeError
        assert isinstance(reldir, str), TypeError
        assert isinstance(name, str), TypeError
        assert isinstance(ext, str), TypeError
        assert isinstance(label, str), TypeError

        self.root = root
        self.reldir = reldir
        self.name = name
        self.ext = ext
        self.label = label

    @property
    def seq(self):
        return self.ext in __SUPPORTED_IMAGES__["RGB"]

    @property
    def absdir(self):
        return os.path.join(self.root, self.reldir)

    @property
    def filename(self):
        if not self.seq:
            if (self.ext != "") and (self.ext is not None):
                return self.name + "." + self.ext
        return self.name

    @property
    def path(self):
        return os.path.join(self.absdir, self.name)

    @property
    def fcount(self):
        self.fcount = None
        if self.seq:
            self.fcount = len(os.listdir(self.path))


        self.seq = ext in __SUPPORTED_IMAGES__[mod]
        self.path = os.path.join(root, reldir)
        self.fcount = None
        if self.seq:
            self.fcount = len(os.listdir(self.path))
        
    def __repr__(self, idents=0):
        string = idents * "\t" + "DataPoint: \n"
        string += idents * "\t" + str(self.name)
        if self.seq:
            string += "(frame sequence len [{}])".format(self.fcount)
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
        
        # both name can be converted to int
        if isinstance(name_0, int) and isinstance(name_1, int):
            return name_0 < name_1
        # compare order: label -> name
        else:
            if self.label != other.label:
                return self.label < other.label
            else:
                return self.name < other.name

    def root_migrated(self, new_root):
        """
        Get a new DataPoint object with dataset root migrated to a new one.
        """
        datapoint = copy.deepcopy(self)
        datapoint.root = new_root
        datapoint.path = os.path.join(datapoint.root,
                                      datapoint.reldir)
        return datapoint


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
            sample.reldir = strip_extension(sample.reldir)
            sample.path = strip_extension(sample.path)
            return sample
        
        def _to_vid(sample, ext):
            """any sample -> video sample
            """
            sample.seq = False
            sample.ext = ext
            sample.reldir = strip_extension(sample.reldir) + "." + ext
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

    # Filter samples using a filter.
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
    a = DataPoint(root="Foo", reldir="Bar", name="test", ext="avi")
    print(a)
    b = a.root_migrated("Fooood")
    print(b)
    c = b.extension_migrated(ext="jpg")
    print(c)
    d = DataPoint(root="Foo", reldir="Bar", name="aha", ext="avi")
    d1000 = DataPoint(root="Foo", reldir="Bar", name="1000", ext="avi")
    d9 = DataPoint(root="Foo", reldir="Bar", name="9", ext="avi")
    print("Test Equality", a == c)
    print("Test Order", a < c, c < a, d < a, d9 < d1000)




if __name__ == "__main__":

    test_sample()
