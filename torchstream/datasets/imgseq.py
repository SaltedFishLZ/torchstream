""" Image Sequences
"""
__all__ = [
    "ImageSequence",
    "ClippedImageSequence",
    "SegmentedImageSequence"
]

import os
import logging

import numpy as np

from . import __config__
from .metadata.sample import Sample, SampleSet
from .utils.vision import frame2ndarray

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



# ------------------------------------------------------------------------- #
#                   Main Classes (To Be Used outside)                       #
# ------------------------------------------------------------------------- #

class ImageSequence(object):
    """A wrapper for a folder containing dumped frames from a video.
    The folder shall looks like this:
    video path
    ├── frame 0
    ├── frame 1
    ├── ...
    └── frame N
    NOTE: Each frame must be named as %d.<ext> (e.g., 233.jpg). Do not
    use any padding zero in the file name! And this class only stores file
    pointers
    NOTE: Following the "do one thing at once" priciple, we only deal with 1 
    data type of 1 data modality in 1 collector object.
    """
    def __init__(self, path, ext, cin="BGR", cout="RGB", **kwargs):
        """
        """
        self.path   =   path
        self.ext    =   ext
        self.cin    =   cin
        self.cout   =   cout
        self.fcount =   0       # frame counnt
        self.fids   =   []      # frame ids

        ## parse kwargs
        if ("tmpl" in kwargs):
            self.tmpl = kwargs["tmpl"]
        else:
            self.tmpl = "{}"
        if ("offset" in kwargs):
            self.offset = kwargs["offset"]
        else:
            self.offset = 0
        
        ## seek all valid frames and add their indices
        _fcnt = 0
        _file_path = self.get_frame_path(_fcnt)
        while (os.path.exists(_file_path)):
            _fcnt += 1         
            _file_path = self.get_frame_path(_fcnt)
        
        if __config__.__STRICT__:
            assert (_fcnt > 0), "Empty video folder {}".format(path)
        else:
            if (0 == _fcnt):
                warn_str = "ImageSequence: [__init__] "
                warn_str += "empty video folder {}".format(path)
                logging.warn(warn_str)

        self.fcount = _fcnt
        self.fids = list(range(_fcnt))

    def get_frame_path(self, idx):
        """
        get the path of idx-th frame
        NOTE: currently, we all use the original frame index
        """
        _filename = self.tmpl.format(idx + self.offset)
        _filename += "." + self.ext
        _filepath = os.path.join(self.path, _filename)
        return(_filepath)

    def get_farray(self, idx):
        """
        get the farray of the idx-th frame
        """
        # generate path & santity check
        assert (idx <= self.fids[self.fcount - 1]), \
            "Image index [{}] exceeds max fid[{}]"\
            .format(idx, self.fids[self.fcount - 1])
        _fpath = self.get_frame_path(idx)
        # call frame2ndarray to get image array
        farray = frame2ndarray(_fpath, self.cin, self.cout)
        # output status
        if __verbose__:
            info_str = "ImageSequence: get_farray success, "
            info_str += "shape "+str(farray.shape)
            logging.info(info_str)
            if __vverbose__:
                print(info_str)
        return(farray)

    def get_varray(self, indices=None):
        """
        get the varray of all the frames, if indices == None.
        otherwise get certain frames as they are a continuous video
        """
        if (indices is None):
            _indices = self.fids
        else:
            # only enable santity check in strict mode for higher perfomance
            if __config__.__STRICT__:
                for idx in indices:
                    assert (idx < self.fcount), "Image index {} overflow".\
                        format(idx)
            _indices = indices
        # generate file paths
        _fpaths = []
        for idx in _indices:
            _fpaths.append(self.get_frame_path(idx))
        # call frames2ndarray to get array
        varray = frames2ndarray(_fpaths, self.cin, self.cout)
        # output status
        if __verbose__:
            info_str = "ImageSequence: get_varray success, "
            info_str += "shape "+str(varray.shape)
            logging.info(info_str)
            if __vverbose__:
                print(info_str)
        return(varray)


class ClippedImageSequence(ImageSequence):
    """
    This class is used to manage clipped video.
    Although you can use data transform to clip an entire video, it has to
    load all frames and select some frames in it. It is not efficient if you
    have preprocessed the video and dump all frames.
    """
    def __init__(self, path, clip_len,
            ext="jpg", cin="BGR", cout="RGB", **kwargs):
        super(ClippedImageSequence, self).__init__(
            path=path, ext=ext,
            cin=cin, cout=cout, **kwargs)
        self.fids = self.__clip__(self.fids, self.fcount, clip_len)
        self.fcount = clip_len

    @staticmethod
    def __clip__(fids, fcount, clip_len):
        """
        __clip__ is made an independent function in case you may need to use
        it in other places
        """
        assert (fcount >= clip_len), \
            "Clip length [{}] exceeds video length [{}]"\
                .format(clip_len, fcount)
        
        if (fcount == clip_len):
            return(fids)
        else:
            # random jitter in time dimension, and re-sample frames
            offset = np.random.randint(fcount - clip_len)
            if __verbose__:
                info_str = "ClippedImageSequence: [__clip__] "
                info_str += "clip frames [{}, {})".\
                    format(offset, offset + clip_len)
                logging.info(info_str)
                if __vverbose__:
                    print(info_str)
            return(fids[offset : offset + clip_len])


class SegmentedImageSequence(ImageSequence):
    """
    This class is used to manage segmented video.
    Although you can use data transform to get a segmented video, it has to
    load all frames and select some frames in it. It is not efficient.
    """
    def __init__(self, path, seg_num,
            ext="jpg", cin="BGR", cout="RGB", **kwargs):
        super(SegmentedImageSequence, self).__init__(
            path=path, ext=ext,
            cin=cin, cout=cout, **kwargs)
        self.fids = self.__segment__(self.fids, self.fcount, seg_num)
        self.fcount = seg_num


    @staticmethod
    def __segment__(frames, fcount, seg_num):
        """
        __segment__ is made an independent function in case you may need to 
        re-use it in other places       
        """
        assert (seg_num > 0), "Segment number must > 0"
        assert (fcount >= seg_num), \
            "Segment number [{}] exceeds video length [{}]".\
                format(seg_num, fcount)

        # interval (length of each segment) = ceil(fcount/seg_num)
        # ((a + b - 1) // b) == ceil(a/b)
        _interval = (fcount + seg_num - 1) // seg_num
        _residual = fcount - _interval * (seg_num - 1)
        _kfids = []         # key frame ids

        # original TSN uses random key frames
        if (_residual == 0):
            for _i in range(seg_num):
                _idx = _i * _interval + np.random.randint(_interval)
                _kfids.append(_idx)
        else:
            for _i in range(seg_num - 1):
                _idx = _i * _interval + np.random.randint(_interval)
                _kfids.append(_idx)
            _idx = _interval * (seg_num - 1) + np.random.randint(_residual)
            _kfids.append(_idx)            

        if __verbose__:
            info_str = "SegmentedImageSequence: [__segment__] "
            info_str += "key frames {}".format(_kfids)
            logging.info(info_str)
            if __vverbose__:
                print(info_str)

        return(_kfids)
