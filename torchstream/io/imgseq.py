""" Image Sequences
"""
import os
import logging
import numpy as np

from . import __config__
from .metadata.datapoint import DataPoint
from .utils.vision import frame2ndarray, frames2ndarray

# configuring logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(__config__.LOGGER_LEVEL)

# ------------------------------------------------------------------------- #
#                   Main Classes (To Be Used outside)                       #
# ------------------------------------------------------------------------- #


class ImageSequence(object):
    """A wrapper for a folder containing dumped frames from a video.
    """
    def __init__(self, datapoint=None, **kwargs):
        """
        """
        fcount = None
        ## parse necessary arguments
        if datapoint is not None:
            assert isinstance(datapoint, DataPoint), TypeError
            assert datapoint.seq, "Not a sequence"
            path = datapoint.path
            ext = datapoint.ext
            fcount = datapoint.fcount
            if datapoint.mod == "RGB":
                cin = "BGR"
                cout = "RGB"
            else:
                raise NotImplementedError
        else:
            assert "path" in kwargs, "Missing parameter [path]"
            assert "ext" in kwargs, "Missing parameter [ext]"
            path = kwargs["path"]
            ext = kwargs["ext"]
            if "mod" in kwargs:
                mod = kwargs["mod"]
                if mod == "RGB":
                    cin="BGR"
                    cout="RGB"
                else:
                    raise NotImplementedError
            else:
                assert "cin" in kwargs, "Missing parameter [cin]"
                assert "cout" in kwargs, "Missing parameter [cout]"
                cin = kwargs["cin"]
                cout = kwargs["cout"]

        self.path = path
        self.ext = ext
        self.cin = cin
        self.cout = cout

        ## parse optional kwargs
        if ("tmpl" in kwargs):
            self.tmpl = kwargs["tmpl"]
        else:
            self.tmpl = "{}"
        if ("offset" in kwargs):
            self.offset = kwargs["offset"]
        else:
            self.offset = 0

        # initialization
        if fcount is None:
            self.fcount = 0       # frame count
            self.fpaths = []      # frame paths
            # seek all valid frames and add their indices
            file_path = self.get_frame_path(self.fcount)
            while (os.path.exists(file_path)):
                self.fpaths.append(file_path)
                self.fcount += 1       
                file_path = self.get_frame_path(self.fcount)
        else:
            self.fcount = fcount
            self.fpaths = [self.get_frame_path(i) for i in range(fcount)]
            

        if __config__.__STRICT__:
            assert self.fcount > 0, "Empty video folder {}".format(path)
            if self.fcount <= 0:
                err_str = "empty video folder {}".format(path)
                logger.error(err_str)

    def get_frame_path(self, idx):
        """get the path of idx-th frame
        Args:
            idx: frame index, from 0
        """
        _filename = self.tmpl.format(idx + self.offset)
        _filename += "." + self.ext
        _filepath = os.path.join(self.path, _filename)
        return _filepath

    def get_farray(self, idx):
        """get the farray of the idx-th frame
        """
        # generate path & santity check
        assert idx < self.fcount, "Frame index [{}] exceeds fcount [{}]".\
            format(idx, self.fcount - 1)
        _fpath = self.fpaths[idx]
        # call frame2ndarray to get image array
        farray = frame2ndarray(_fpath, self.cin, self.cout)
        # logging
        info_str = "ImageSequence: [get_farray] success, "
        info_str += "shape "+str(farray.shape)
        logger.info(info_str)
        # return
        return farray

    def get_varray(self, indices=None):
        """get the varray of all the frames, if indices == None.
        otherwise get certain frames as they are a continuous video
        """
        # use global file paths
        if (indices is None):
            _fpaths = self.fpaths
        # generate file paths
        else:
            _fpaths = []
            for _idx in indices:
                _fpaths.append(self.get_frame_path(_idx))
        # call frames2ndarray to get array
        varray = frames2ndarray(_fpaths, self.cin, self.cout)
        # logging
        info_str = "ImageSequence: get_varray success, "
        info_str += "shape "+str(varray.shape)
        logger.info(info_str)
        # return
        return varray

    def __array__(self):
        """Numpy interface
        """
        return self.get_varray()


def _to_imgseq(x, **kwargs):
    """
    """
    assert isinstance(x, DataPoint), TypeError
    return ImageSequence(x, **kwargs)
