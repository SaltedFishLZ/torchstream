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

from . import __config__
from .datapoint import DataPoint
from .backends.opencv import video2frames, frames2ndarray, ndarray2video

try:
    from subprocess import DEVNULL  # python 3.x
except ImportError:
    DEVNULL = open(os.devnull, "wb")

# configuring logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(__config__.LOGGER_LEVEL)


def vid2vid(src, dst, **kwargs):
    """Video -> Video Conversion
    using ffmpeg to convert videos types
    Args:
        src (DataPoint): source video's meta-data
        dst (DataPoint): destination video's meta-data
    Optional:
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
        assert not dst.seq, \
            "destination sample is not a video"

    success = False
    fails = 0
    retries = 0
    retries = kwargs.get("retries", 0)

    src_vid = src.path
    dst_vid = dst.path

    dst_dir = os.path.dirname(dst_vid)
    os.makedirs(dst_dir, exist_ok=True)

    command = " ".join(["ffmpeg", "-i",
                        "\"{}\"".format(src_vid),
                        "\"{}\"".format(dst_vid),
                        "-y"])

    while fails <= retries:
        _subp = subprocess.run(command, shell=True, check=False,
                               stdout=DEVNULL, stderr=DEVNULL)
        if _subp.returncode == 0:
            success = True
            break
        else:
            fails += 1
    return success


def vid2seq(src, dst, **kwargs):
    """Video -> Image Sequence Conversion
    using io.backends to slicing videos
    Args:
        src (DataPoint): source video's meta-data
        dst (DataPoint): destination video's meta-data
    Optional:
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

    while fails <= retries:
        ret = video2frames(src_vid, dst_seq)
        success = ret[0]
        if success:
            break
        else:
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
