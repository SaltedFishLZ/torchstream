""" Video I/O Conversion

NOTE:
vid2vid requires `ffmpeg`
"""
__all__ = [
    "vid2vid", "vid2seq", "seq2vid"
]

import os
import logging
import subprocess

from torchstream.transforms.functional import resize
from . import __config__
from .datapoint import DataPoint
from .backends.opencv import video2ndarray, video2frames,\
    frames2ndarray, ndarray2video, ndarray2frames

try:
    from subprocess import DEVNULL  # python 3.x
except ImportError:
    DEVNULL = open(os.devnull, "wb")

# configuring logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(__config__.LOGGER_LEVEL)


def vid2vid(src, dst, backend="ffmpeg", **kwargs):
    """Video -> Video Conversion
    using ffmpeg/OpenCV to convert videos types
    Args:
        src (DataPoint): source video's meta-data
        dst (DataPoint): destination video's meta-data
        backend (str): video2video conversion backend,
            options:
                "ffmpeg" (via bash commands),
                "opencv" (via opencv-python),
    Optional:
        retries (int): retry number
        fps (int): target video's fps, valid for ffmpeg
        scale (float): resolution scale between target video and
            source video (e.g., Ho/Hi). We keep the aspect ratio. 
    """
    if __config__.STRICT:
        # check source operand
        assert isinstance(src, DataPoint), \
            TypeError
        assert not src.seq, \
            "source sample is not a video"
        # check destination operand
        assert isinstance(dst, DataPoint), \
            TypeError
        assert not dst.seq, \
            "destination sample is not a video"
    assert backend in ["ffmpeg", "opencv"]

    success = False
    fails = 0
    retries = 0
    retries = kwargs.get("retries", 0)

    src_vid = src.path
    dst_vid = dst.path

    dst_dir = os.path.dirname(dst_vid)
    os.makedirs(dst_dir, exist_ok=True)

    # branch 0
    # call ffmpeg in bash to convert video directly
    if backend == "ffmpeg":
        # assemble command
        # NOTE: add trailing space!
        output_options = " -strict experimental "

        # fps
        if "fps" in kwargs:
            fps = kwargs["fps"]
            output_options += " -r {} ".format(fps)

        # resolution scaling
        if "scale" in kwargs:
            scale = kwargs["scale"]
            opt_fmt_str = " -vf \"scale=iw*{}:ih*{}\" "
            output_options += opt_fmt_str.format(scale, scale)

        # bash command final stage
        command = " ".join(["ffmpeg", "-i",
                            "\"{}\"".format(src_vid),
                            output_options,
                            "\"{}\"".format(dst_vid),
                            "-y"])

        # run bash command
        while fails <= retries:
            _subp = subprocess.run(command, shell=True, check=False,
                                   stdout=DEVNULL, stderr=DEVNULL)
            if _subp.returncode == 0:
                success = True
                break
            else:
                fails += 1

    # branch 1
    # call torchstream opencv lib in python
    # video -> varray -> video
    elif backend == "opencv":

        def convert(src_vid, dst_vid, **kwargs):
            """Local conversion function
            Args:
                src_vid(str): path to source video
                dst_vid(str): path to desination video
            Optional:
                scale(float): resolution scale between target video and
                source video (e.g., Ho/Hi). We keep the aspect ratio.
            Return:
                success(bool)
            """
            # TODO: recover fps information
            varray = video2ndarray(src_vid)
            if varray is None:
                return False
            if "scale" in kwargs:
                scale = kwargs["scale"]
                t, h, w, c = varray.shape
                oh, ow = int(scale * h), int(scale * w)
                varray = resize(varray, (oh, ow))
            success = ndarray2video(varray, dst_path=dst_vid)
            return success

        while fails <= retries:
            success = convert(src_vid, dst_vid)
            if success:
                break
            else:
                fails += 1
    else:
        raise NotImplementedError

    return success


def vid2seq(src, dst, transform=None, **kwargs):
    """Video -> Image Sequence Conversion
    using io.backends to slicing videos
    Args:
        src (DataPoint): source video's meta-data
        dst (DataPoint): destination video's meta-data
    Optional:
        transform (callale): varray transformation
        retries: retry number
    """
    if __config__.STRICT:
        # check source operand
        assert isinstance(src, DataPoint), \
            TypeError
        assert not src.seq, \
            "source sample is not a video"
        # check destination operand
        assert isinstance(dst, DataPoint), \
            TypeError
        assert dst.seq, \
            "destination sample is not an image sequence"

    success = False
    fails = 0
    retries = kwargs.get("retries", 0)

    src_vid = src.path
    dst_seq = dst.path
    os.makedirs(dst_seq, exist_ok=True)

    if transform is None:
        while fails <= retries:
            ret = video2frames(src_vid, dst_seq)
            success = ret[0]
            if success:
                break
            fails += 1
    else:
        assert callable(transform), TypeError
        while fails <= retries:
            varray = video2ndarray(src_vid)
            if varray is None:
                fails += 1
                continue

            varray = transform(varray)
            if varray is None:
                fails += 1
                continue

            ret = ndarray2frames(varray, dst_seq)
            success = ret[0]
            if success:
                break

            fails += 1        

    return success


def seq2vid(src, dst, **kwargs):
    """Image Sequence to Video
    Args:
        src (DataPoint): source sequence's meta-data
        dst (DataPoint): destination video's meta-data
    Optional:
        retries: retry number
    """
    if __config__.STRICT:
        # check source operand
        assert isinstance(src, DataPoint), \
            TypeError
        assert src.seq, \
            "source sample is not a image sequence"
        # check destination operand
        assert isinstance(dst, DataPoint), \
            TypeError
        assert not dst.seq, \
            "destination sample is not a video"

    framepaths = src.framepaths
    varray = frames2ndarray(framepaths)

    dst_vid = dst.path
    dst_dir = os.path.dirname(dst_vid)
    os.makedirs(dst_dir, exist_ok=True)

    ndarray2video(varray, dst_path=dst_vid)

    # TODO: error report
    # currently, always return True, pretend always succeed
    return True
