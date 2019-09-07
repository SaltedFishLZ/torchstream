"""Abstract Datapoint Class
"""
import os
import logging

from . import __config__
from .__support__ import SUPPORTED_IMAGES

# configuring logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(__config__.LOGGER_LEVEL)

# constants
UNKNOWN_LABEL = None

class DataPoint(object):
    """Meta-data of a video sample in a certain dataset
    Args:
        root (str): root path of the dataset
        reldir (str): relative directory path of the sample in the dataset
        name (str): file name of the sample (without any extension and path)
        ext (str): file extension of the sample (e.g., "avi", "mp4").
            NOTE: '.' excluded.
        fpath_offset (int): frame number offset
        fpath_tmpl (str): frame name template (e.g., some datasets name frames
        as 00001.jpg, 00002.jpg..., so the fpath_offset is 1, and fpath_tmpl is
        {:05d})
        label (str): class label of this sample
    """
    def __init__(self, root, reldir, name, ext="jpg",
                 fpath_offset=0, fpath_tmpl="{}",
                 label=UNKNOWN_LABEL):
        assert isinstance(root, str), TypeError
        assert isinstance(reldir, str), TypeError
        assert isinstance(name, str), TypeError
        assert isinstance(ext, str), TypeError
        if label != UNKNOWN_LABEL:
            assert isinstance(label, str), TypeError

        # scan the file system to get mroe information
        # NOTE: need to check compatibility each time update
        # these entries

        self.root = root
        self.reldir = reldir
        self.name = name
        self.ext = ext
        self.fpath_offset = fpath_offset
        self.fpath_tmpl = fpath_tmpl
        self.label = label

        # cache attributes
        self._seq = self.seq
        self._path = self.path
        self._fcount = self.fcount

    @property
    def seq(self):
        """
        Return:
            bool, whether this datapoint is a sequence or not.
        """
        return self.ext in SUPPORTED_IMAGES["RGB"]

    @property
    def absdir(self):
        """
        Return:
            str, absolute path of the directory where this datapoint
            lies.
        """
        return os.path.join(self.root, self.reldir)

    @property
    def filename(self):
        """
        Return: str, file name with extension for videos,
            sequence folder name for image sequence.
        """
        if not self.seq:
            if (self.ext != "") and (self.ext is not None):
                return self.name + "." + self.ext
        return self.name

    @property
    def path(self):
        """
        Return: absolute file path for videos, absolute
            folder path for image sequence.
        """
        return os.path.join(self.absdir, self.filename)

    @property
    def fcount(self):
        """
        Return:
            None for videos;
            # frames for image sequence;
        """
        if self.seq:
            # TODO: os.scandir compatibility
            # TODO: check valid files, bypass invalid files
            return(len(os.listdir(self.path)))
        return -1

    @property
    def framepaths(self):
        """
        Return:
            list (str), frame path string with extension
        """
        assert self.seq, TypeError("Only image sequences have framepaths")

        framepaths = []
        for _i in range(self.fcount):
            fpath = self.fpath_tmpl.format(_i + self.fpath_offset)
            fpath += "." + self.ext
            fpath = os.path.join(self.path, fpath)
            framepaths.append(fpath)
        return framepaths

    def __repr__(self, idents=0):
        string = idents * "\t" + "DataPoint: \n"
        string += idents * "\t" + str(self.name)
        if self.seq:
            string += "(frame sequence len [{}])".format(self.fcount)
        string += '\n'
        string += idents * "\t" + "[name] : {}  \t".format(self.name)
        string += idents * "\t" + "[label] : {}  \t".format(self.label)
        string += idents * "\t" + "[path] : {}".format(self.path)
        return string

    def __eq__(self, other):
        if isinstance(other, DataPoint):
            for _k in self.__dict__:
                if (self.__dict__[_k] != other.__dict__[_k]):
                    return False
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        NOTE: comparison between 2 datapoints with only file extension being
            different is an undefined behavior.
        """
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
        # comparison order: label -> name
        else:
            if self.label != other.label:
                return self.label < other.label
            else:
                return self.name < other.name
