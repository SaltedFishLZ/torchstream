""" Image Sequences
"""
import os
import logging

from . import __config__
from .datapoint import DataPoint
from .backends.opencv import frame2ndarray, frames2ndarray

# configuring logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(__config__.LOGGER_LEVEL)


class ImageSequence(object):
    """A wrapper for a folder containing dumped frames from a video.
    Currently, only support RGB/BGR input.
    """
    def __init__(self, datapoint, **kwargs):
        """
        """
        assert isinstance(datapoint, DataPoint), TypeError
        assert datapoint.seq, TypeError("Not a sequence!")
        self.datapoint = datapoint

        # frame name template
        self.tmpl = "{}"
        if ("tmpl" in kwargs):
            self.tmpl = kwargs["tmpl"]

        # frame index offset
        self.offset = 0
        if ("offset" in kwargs):
            self.offset = kwargs["offset"]

        # frame paths
        self.fpaths = []
        for _idx in range(self.fcount):
            _fpath = self.get_frame_path(_idx)
            self.fpaths.append(_fpath)

        if __config__.STRICT:
            assert self.fcount > 0, "Empty video folder {}".format(self.path)

    @property
    def path(self):
        return self.datapoint.path

    @property
    def fcount(self):
        return self.datapoint.fcount

    @property
    def ext(self):
        return self.datapoint.ext

    def get_frame_path(self, idx):
        """get the absolute path of idx-th frame
        Args:
            idx: frame index, from self.offset
        """
        _filename = self.tmpl.format(idx + self.offset)
        _filename += "." + self.ext
        _filepath = os.path.join(self.path, _filename)
        return _filepath

    def get_farray(self, idx):
        """get the farray of the idx-th frame
        """
        # generate path
        assert idx < self.fcount, "Frame index [{}] exceeds fcount [{}]".\
            format(idx, self.fcount - 1)
        _fpath = self.fpaths[idx]

        farray = frame2ndarray(_fpath)

        # logging
        info_str = "[get_farray] success, shape {}".format(farray.shape)
        logger.info(info_str)

        return farray

    def get_varray(self, indices=None):
        """get the varray of all the frames, if indices == None.
        otherwise get certain frames as they are a continuous video
        """
        # use global file paths or slicing specified paths
        if (indices is None):
            _fpaths = self.fpaths
        else:
            _fpaths = []
            for _idx in indices:
                _fpaths.append(self.get_frame_path(_idx))

        varray = frames2ndarray(_fpaths)

        # logging
        info_str = "[get_varray] success, shape {}".format(varray.shape)
        logger.info(info_str)

        return varray

    def __array__(self):
        """Numpy interface
        """
        return self.get_varray()
