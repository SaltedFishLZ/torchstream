"""Video Analysis Tools
"""
import os

import numpy as np

from .datapoint import DataPoint
from .backends import opencv as backend

try:
    from subprocess import DEVNULL  # python 3.x
except ImportError:
    DEVNULL = open(os.devnull, "wb")

# ---------------------------------------------------------------- #
#                       Pixel Level Statistics                     #
# ---------------------------------------------------------------- #


def varray_sum(varray, **kwargs):
    """
    Get the sum of all pixels in a varray for each channel

    Args:
        varray (np.ndarray): input video blob

    Return:
        tuple: (sum array, pixel number array), we use 1d np array
            for multi-channels.
    """
    # get frame shape, leave the channel dimension alone
    (_t, _h, _w) = varray.shape[0:3]
    nums = _t * _h * _w

    # Numpy sum over multiple axes
    # reference:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    sums = varray.sum(axis=(0, 1, 2))

    return (sums, nums)


def varray_rss(varray, means=None, **kwargs):
    """Get the RSS(residual sum of squares) of all pixels in a varray of
    each channel.

    Args:
        varray (np.ndarray): input blob
        means (np.ndarray): means of the varray. If not specified, this
        function will call varray_sum_raw to calculate it.

    Return:
        tuple: (rss array, pixel numbers)

    Reference:
        https://en.wikipedia.org/wiki/Residual_sum_of_squares
    """
    if means is None:
        sums, nums = varray_sum(varray)
        means = sums / nums

    (_t, _h, _w) = varray.shape[0:3]
    nums = _t * _h * _w
    # # DEBUG
    # print(means)
    residuals = varray - np.tile(means, (_t, _h, _w, 1))
    residual_squares = np.square(residuals)

    # Numpy sum over multiple axes
    # reference:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    rsses = residual_squares.sum(axis=(0, 1, 2))

    return (rsses, nums)


def datapoint_sum(datapoint, frame_sampler=None):
    """Wrapper for DataPoint
    """
    assert isinstance(datapoint, DataPoint), TypeError
    if datapoint.seq:
        loader = backend.frames2ndarray
        fpaths = datapoint.framepaths
        if frame_sampler is not None:
            fpaths = frame_sampler(fpaths)
        varray = loader(fpaths)
    else:
        loader = backend.video2ndarray
        path = datapoint.path
        varray = loader(path)
    return varray_sum(varray)


def datapoint_rss(datapoint, frame_sampler=None):
    """Wrapper for DataPoint
    """
    assert isinstance(datapoint, DataPoint), TypeError
    if datapoint.seq:
        loader = backend.frames2ndarray
        fpaths = datapoint.framepaths
        if frame_sampler is not None:
            fpaths = frame_sampler(fpaths)
        varray = loader(fpaths)
    else:
        loader = backend.video2ndarray
        path = datapoint.path
        varray = loader(path)
    return varray_rss(varray)

# ---------------------------------------------------------------- #
#                       Frame Level Statistics                     #
# ---------------------------------------------------------------- #


def varray_len(varray):
    """Number of frames in a varray
    Args:
        varray (np.ndarray): input video blob
    Return:
        int
    """
    return varray.shape[0]


def varray_hxw(varray):
    """ (H, W) of a varray
    Args:
        varray (np.ndarray): input video blob
    Return:
        (int, int)
    """
    return varray.shape[1:3]


def datapoint_len(datapoint, check_fn=None):
    """Wrapper for DataPoint
    Args:
        check_fn (callable)
    """
    assert isinstance(datapoint, DataPoint), TypeError

    if datapoint.seq:
        datapoint_len = datapoint.fcount
    else:
        loader = backend.video2ndarray
        path = datapoint.path
        varray = loader(path)
        datapoint_len = varray_len(varray)

    if check_fn is not None:
        check_fn(datapoint_len)

    return datapoint_len


def datapoint_hxw(datapoint, **kwargs):
    """Wrapper for DataPoint
    """
    assert isinstance(datapoint, DataPoint), TypeError

    if datapoint.seq:
        loader = backend.frame2ndarray
        fpaths = datapoint.framepaths
        assert len(fpaths) > 0, ValueError("Empty image sequence")
        farray = loader(fpaths[0])
        hxw = farray.shape[0:2]
    else:
        loader = backend.video2ndarray
        path = datapoint.path
        varray = loader(path)
        hxw = varray_hxw(varray)

    return hxw


def test_datapoint_fps():
    file_name = "Climbing_roof_in_TCA_climb_f_cm_np1_ba_med_2.avi"
    datapoint = DataPoint(root="~/Datasets/HMDB51/HMDB51-avi/",
                          rpath="climb/{}".format(file_name),
                          name=file_name,
                          ext="avi")
    print(datapoint_fps(datapoint))

