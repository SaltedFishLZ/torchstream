## @package metadata
# Dataset Meta-data Management Module
#
#

import os
import copy
import logging

from . import __config__
from .__const__ import IMGSEQ, UNKOWN_LABEL, UNKOWN_CID
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




## Class
#  
class Sample(object):
    """
    An video sample struct containing the meta-data of a video sample
    """

    ## Constructor
    #  
    #  @param root str: absolute root path of the dataset
    #  @param path str: absolute path of 1 video sample
    #  @param name str: file name (without any extension and path)
    #  @param ext str:  file extension (e.g., "avi", "mp4"), '.' excluded.  
    #      If it == constant.IMGSEQ, it means the sample is sliced into a 
    #      sequence of images (not limited to RGB modality) and the images 
    #      are stored in the folder which is the "path" mentioned above.
    #      NOTE: Currently, we only handle images stored in .jpg formats
    #  @param lbl str:  
    #      label of the sample, is a unique string in certain dataset
    #  @param cid str:  
    #      class id of the sample, is the numerical representation of label
    def __init__(self, root, rpath, name,
                 mod="RGB", ext=IMGSEQ,
                 lbl=UNKOWN_LABEL, cid=UNKOWN_CID):
        """
        Initailization function
        """
        # Santity Check
        if __config__.__STRICT__:
            assert mod in __SUPPORTED_MODALITIES__, NotImplementedError
            assert ext in __SUPPORTED_MODALITIES__[mod], NotImplementedError
        
        self.root = root
        self.rpath = rpath
        self.name = name

        self.mod = mod
        self.ext = ext
        self.seq = ext in __SUPPORTED_IMAGES__[mod]

        self.path = os.path.join(root, rpath)

        self.lbl = lbl
        self.cid = cid

    ## Documentation for a method.
    #  @param self The object pointer.
    def __repr__(self, idents=0):
        string = idents * "\t" + "Sample Object: \n"
        string += idents * "\t" + str(self.name)
        if self.seq:
            string += "(frame sequence)"
        string += '\n'
        string += idents * "\t" + "[label] : {}  \t".format(self.lbl)
        string += idents * "\t" + "[cid] : {} \t".format(self.cid)
        string += idents * "\t" + "[path] : {}".format(self.path)
        return string

    ## Documentation for a method.
    #  @param self The object pointer.
    def __eq__(self, other):
        if isinstance(other, Sample):
            return(
                (self.root == other.root)
                and (self.path == other.path)
                and (self.name == other.name)
                and (self.mod == other.mod)
                and (self.ext == other.ext)
                and (self.seq == other.seq)
                and (self.lbl == other.lbl)
                and (self.cid == other.cid)
            )
        return False

    ## Documentation for a method.
    #  @param self The object pointer.
    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        Sorting samples according to samples' names.
        If sample name can be castd to int, use int for
        comparison.
        """
        ## santity check
        assert isinstance(other, Sample), TypeError
        
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
            if self.lbl != other.lbl:
                return self.lbl < other.lbl
            else:
                return self.name < other.name

    ## Documentation for a method.
    #  @param self The object pointer.
    def __hash__(self):
        return(hash((
            self.root,
            self.path,
            self.name,
            self.mod,
            self.ext,
            self.seq,
            self.lbl,
            self.cid
        )))

    ## Documentation for a method.
    #  @param self The object pointer.    
    def root_migrated(self, new_root):
        """
        Get a new Sample object with dataset root migrated to a new one.
        """
        return(
            Sample(
                root=new_root,
                rpath=self.rpath,
                name=self.name,
                mod=self.mod,
                ext=self.ext,
                lbl=self.lbl,
                cid=self.cid
            )
        )


    def extension_migrated(self, ext):
        """
        Get a new Sample object with file extension migrated
        modality must be the same.
        """
        assert ext in __SUPPORTED_MODALITIES__[self.mod], NotImplementedError
        _sample = copy.deepcopy(self)
        
        def _to_seq(sample, ext):
            """
            any sample -> sequence sample
            no santity check
            """
            sample.seq = True
            sample.ext = ext
            sample.rpath = strip_extension(sample.rpath)
            sample.path = strip_extension(sample.path)
            return sample
        
        def _to_vid(sample, ext):
            """
            any sample -> video sample
            no santity check
            """
            sample.seq = False
            sample.ext = ext
            sample.rpath = strip_extension(sample.rpath) + "." + ext
            sample.path = strip_extension(sample.path) + "." + ext
            return sample

        if ext in __SUPPORTED_IMAGES__[self.mod]:
            return _to_seq(_sample, ext)
        elif ext in __SUPPORTED_VIDEOS__[self.mod]:
            return _to_vid(_sample,ext)
        else:
            raise NotImplementedError


## SampleSet: A class containing a set of samples with some statistics
#  
#  Details :
class SampleSet(set):
    """A set of samples with some statistics information
    
    Args:
        samples (set): a set of Sample objects
        counts (set): sample counts for each label
    """
    ##  Constructor Function
    #  @param samples set: a set of Sample objects.
    #  @param labels set|list: a set/list of label names (str).
    #  If not specified, will count all possible labels from the samples.
    #  @param eager bool: eager execution
    def __init__(self, samples, labels=None):
        """
        """
        # santity check
        assert isinstance(samples, set), TypeError

        super(SampleSet, self).__init__(samples)
        
        self.counts = dict()
        ## no label limitation
        if labels is None:
            for _sample in samples:
                _label = _sample.lbl
                if _label in self.counts:
                    self.counts[_label] += 1
                else:
                    self.counts[_label] = 1
        ## has label limitation
        else:
            self.counts = dict.fromkeys(labels, 0)
            for _sample in samples:
                _label = _sample.lbl
                if _label in labels:
                    self.counts[_label] += 1
                    super(SampleSet, self).add(_sample)

    ## Documentation for a method.
    #  @param self The object pointer.
    def __repr__(self, idents=0):
        string = idents * "\t" + "SampleSet Object: \n"
        for _label in self.counts:
            string += idents * "\t" + "[{}]: \t{}\n".\
                format(_label, self.counts[_label])
        return string

    def get_samples(self):
        """
        Get the samples set
        """
        return set(self)

    def get_statistics(self):
        """
        Get the statistics
        """
        return self.counts

    ## Filter samples using a filter.
    #  @param filter_ callable: a filter with input being a Sample object
    def filter_samples(self, filter_):
        """Remove samples according to a given filter

        Args:
            filter_ (function): A filter function taking 1 Sample object as
                input and return whether the sample should be preserved (True
                or False)
        """
        _samples = set()
        for _sample in self:
            if not filter_(_sample):
                info_str = "SampleSet:[filter_samples], remove\n{}"\
                        .format(_sample)
                logger.info(info_str)
            else:
                _samples.add(_sample)

        self.__init__(_samples)

    ## Documentation for a method.
    #  @param self The object pointer.
    def update_labels(self, labels):
        """
        Update labels and corresponding counts. 
        """
        for _label in labels:
            if _label == UNKOWN_LABEL:
                continue
            # set new label's statistics
            if _label not in self.counts:
                self.counts[_label] = 0

    ## Documentation for a method.
    #  @param self The object pointer.
    def update_samples(self, samples):
        """
        Update self.samples (add new samples) and corresponding statistics.
        NOTE: currently, we only update thoses samples with the same labels
        of existing ones.

        @param samples set: a set of Sample objects.
        """
        labels = self.counts.keys()
        for _sample in samples:
            _label = _sample.lbl
            if _label == UNKOWN_LABEL:
                continue
            if _label in labels:
                self.counts[_label] += 1
                super(SampleSet, self).add(_sample)  

    ## Documentation for a method.
    #  @param self The object pointer.
    def refresh_statistics(self):
        """!
        Update self.counts to make all statstistics consistent when you change
        the data without using member functions
        """
        self.counts = dict.fromkeys(self.counts.keys(), 0)
        for _sample in self.get_samples():
            _label = _sample.lbl
            if _label in self.counts:
                self.counts[_label] += 1
                super(SampleSet, self).add(_sample)




# ------------------------------------------------------------------------- #
#              Self-test Utilities (Not To Be Used outside)                 #
# ------------------------------------------------------------------------- #

def test_sample():
    a = Sample(root="Foo", rpath="Bar", name="test", ext="avi")
    print(a)
    b = a.root_migrated("Fooood")
    print(b)
    c = b.extension_migrated(ext="jpg")
    print(c)
    d = Sample(root="Foo", rpath="Bar", name="aha", ext="avi")
    d1000 = Sample(root="Foo", rpath="Bar", name="1000", ext="avi")
    d9 = Sample(root="Foo", rpath="Bar", name="9", ext="avi")
    print("Test Equality", a == c)
    print("Test Order", a < c, c < a, d < a, d9 < d1000)




if __name__ == "__main__":

    test_sample()