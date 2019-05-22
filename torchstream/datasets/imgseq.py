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
from .metadata.sample import Sample
from .utils.vision import frame2ndarray, frames2ndarray

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

# ---------------------------------------------------------------- #
#                  Configuring Python Logger                       #
# ---------------------------------------------------------------- #

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
if __config__.__VERY_VERY_VERBOSE__:
    logger.setLevel(logging.INFO)
elif __config__.__VERY_VERBOSE__:
    logger.setLevel(logging.WARNING)
elif __config__.__VERBOSE__:
    logger.setLevel(logging.ERROR)
else:
    logger.setLevel(logging.CRITICAL)



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
    NOTE: Following the "do one thing at once" priciple, we only deal with 1 
    data type of 1 data modality in 1 collector object.
    """
    def __init__(self, sample_like=None, **kwargs):
        """
        """
        ## parse necessary arguments
        if sample_like is not None:
            assert isinstance(sample_like, Sample), TypeError
            assert sample_like.seq, "Not a sequence"
            path = sample_like.path
            ext = sample_like.ext
            if sample_like.mod == "RGB":
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

        ## initialization
        self.fcount = 0       # frame count
        self.fpaths = []      # frame paths

        ## seek all valid frames and add their indices
        _file_path = self.get_frame_path(self.fcount)
        while (os.path.exists(_file_path)):
            self.fpaths.append(_file_path)
            self.fcount += 1       
            _file_path = self.get_frame_path(self.fcount)
        
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

class ClippedImageSequence(ImageSequence):
    """ImageSeqence for clipped video
    Although you can use data transform to clip an entire video, it has to
    load all frames and select some frames in it. It is not efficient if you
    have preprocessed the video and dump all frames.
    """
    def __init__(self, sample_like=None, clip_len=1, volatile=True, **kwargs):
        """
        """
        assert clip_len > 0, "clip_len must > 0"
        super(ClippedImageSequence, self).__init__(sample_like, **kwargs)
        self.clip_len = clip_len
        self.volatile = volatile
        self.clip_idxs = None

    @staticmethod
    def __clip__(fcount, clip_len, **kwargs):
        """
        __clip__ is made an independent function in case you may need to use
        it in other places
        Return:
            a list of indices
        """
        assert fcount >= clip_len, \
            "Clip length [{}] exceeds video length [{}]"\
                .format(clip_len, fcount)
        # short path
        if fcount == clip_len:
            return list(range(clip_len))
        # random jitter in time dimension, and re-sample frames
        offset = np.random.randint(fcount - clip_len)
        info_str = "[__clip__] clip frames [{}, {})".\
            format(offset, offset + clip_len)
        logger.info(info_str)
        # get results
        return list(range(offset, offset + clip_len))

    def get_varray(self):
        """
        """
        if self.volatile:
            self.clip_idxs = self.__clip__(self.fcount, self.clip_len)
        else:
            if self.clip_idxs is None:
                self.clip_idxs = self.__clip__(self.fcount, self.clip_len)
        return ImageSequence.get_varray(self, indices=self.clip_idxs)


class SegmentedImageSequence(ImageSequence):
    """ImageSeqence for segmented video
    This class is used to manage segmented video.
    Although you can use data transform to get a segmented video, it has to
    load all frames and select some frames in it. It is not efficient.
    """
    def __init__(self, sample_like=None, seg_num=1, volatile=True, **kwargs):
        """
        """
        assert seg_num > 0, "seg_num must > 0"
        super(SegmentedImageSequence, self).__init__(sample_like, **kwargs)
        self.seg_num = seg_num
        self.volatile = volatile
        self.snip_idxs = None

    @staticmethod
    def __segment__(fcount, seg_num):
        """
        __segment__ is made an independent function in case you may need to 
        re-use it in other places
        Return:
            a list of indices
        """
        assert fcount >= seg_num, \
            "Segment number [{}] exceeds video length [{}]".\
                format(seg_num, fcount)

        # interval (length of each segment) = ceil(fcount/seg_num)
        # ((a + b - 1) // b) == ceil(a/b)
        _interval = (fcount + seg_num - 1) // seg_num
        _residual = fcount - _interval * (seg_num - 1)
        _snip_idxs = []         # key frame ids

        # short path
        if _residual == 0:
            for _i in range(seg_num):
                _idx = _i * _interval + np.random.randint(_interval)
                _snip_idxs.append(_idx)
        else:
            for _i in range(seg_num - 1):
                _idx = _i * _interval + np.random.randint(_interval)
                _snip_idxs.append(_idx)
            _idx = _interval * (seg_num - 1) + np.random.randint(_residual)
            _snip_idxs.append(_idx)

        # logging
        logger.info("[__segment__] frames {}".format(_snip_idxs))

        return _snip_idxs

    def get_varray(self):
        """
        """
        if self.volatile:
            self.snip_idxs = self.__segment__(self.fcount, self.seg_num)
        else:
            if self.snip_idxs is None:
                self.snip_idxs = self.__segment__(self.fcount, self.seg_num)
        return ImageSequence.get_varray(self, indices=self.snip_idxs)












def TestImageSequence():
    test_video = os.path.join(DIR_PATH, "test.avi")
    test_frames = os.path.join(DIR_PATH, "test_frames")
    from .utils.vision import video2frames, farray_show
    video2frames(test_video, test_frames)

    imgseq_0 = ImageSequence(path=test_frames,
                             ext="jpg", cin="BGR", cout="RGB"
                             )

    varray = imgseq_0.get_varray()
    print(varray.shape)
    # print(imgseq_0.get_farray(0).shape)
    # farray_show(caption="test", farray=farray)

    # import cv2
    # (cv2.waitKey(0) & 0xFF == ord("q"))
    # cv2.destroyAllWindows()

    import importlib

    dataset = "weizmann"
    metaset = importlib.import_module(
        "datasets.metadata.metasets.{}".format(dataset))

    kwargs = {
        "root" : metaset.JPG_DATA_PATH,
        "layout" : metaset.__layout__,
        "lbls" : metaset.__LABELS__,
        "mod" : "RGB",
        "ext" : "jpg",
    }
    
    from .metadata.collect import collect_samples
    samples = collect_samples(**kwargs)

    for _sample in samples:
        _imgseq = ImageSequence(_sample)
        print(np.all(np.array(_imgseq) == _imgseq.get_varray()))

def TestClippedImageSequence():
    """
    """
    test_video = os.path.join(DIR_PATH, "test.avi")
    test_frames = os.path.join(DIR_PATH, "test_frames")
    from .utils.vision import video2frames, farray_show
    video2frames(test_video, test_frames)    

    CLIP_LEN = 16

    imgseq_0 = ClippedImageSequence(path=test_frames, clip_len=CLIP_LEN,
                             ext="jpg", cin="BGR", cout="RGB"
                             )
    print(np.array(imgseq_0).shape)

    print("[Volatile]", imgseq_0.volatile)
    offsets = []
    MAX_OFFSET = imgseq_0.fcount - CLIP_LEN
    import tqdm
    for _i in tqdm.tqdm(range(10 * MAX_OFFSET)):
        imgseq_0.get_varray()
        offsets.append(imgseq_0.clip_idxs[0])
    hist = np.histogram(offsets, bins=list(range(MAX_OFFSET+1)))
    print(hist[0])
    
    print("-"*80)
    imgseq_0.volatile = False
    print("[Volatile]", imgseq_0.volatile)    
    for _i in range(20):
        imgseq_0.get_varray()
        print(imgseq_0.clip_idxs)

def TestSegmentedImageSequence():
    """
    """
    test_video = os.path.join(DIR_PATH, "test.avi")
    test_frames = os.path.join(DIR_PATH, "test_frames")
    from .utils.vision import video2frames, farray_show
    video2frames(test_video, test_frames)    

    SEG_NUM = 16

    imgseq_0 = SegmentedImageSequence(path=test_frames, seg_num=SEG_NUM,
                             ext="jpg", cin="BGR", cout="RGB"
                             )
    print(np.array(imgseq_0).shape)

    print("[Volatile]", imgseq_0.volatile)
    _snip_idxs = []
    import tqdm
    for _i in range(10):
        imgseq_0.get_varray()
        print(imgseq_0.snip_idxs)
    # hist = np.histogram(offsets, bins=list(range(MAX_OFFSET+1)))
    # print(hist[0])
    
    print("-"*80)
    imgseq_0.volatile = False
    print("[Volatile]", imgseq_0.volatile)    
    for _i in range(10):
        imgseq_0.get_varray()
        print(imgseq_0.snip_idxs)




if __name__ == "__main__":
    TestSegmentedImageSequence()