import os
import sys
import copy
import logging
import importlib
import multiprocessing as mp

import numpy as np

from .dataset import VideoDataset


def varray_sum_raw(varray):
    (_t, _h, _w) = varray.shape[0:3]
    nums = _t * _h * _w

    # Numpy sum over multiple axes
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    sums = varray.sum(axis=(0,1,2))
    print(sums.shape)

    return(sums, nums)

def varray_sum_rsq(varray, means):
    (_t, _h, _w) = varray.shape[0:2]
    nums = _t * _h * _w

    residuals = varray - means.tile(means, (_t, _h, _w, 1))
    rsquares = np.square(residuals)
    # Numpy sum over multiple axes
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    sums = rsquares.sum(axis=(0,1,2))

    return(sums, nums)


def test_functions():

    from video import video2ndarray
    dir_path = os.path.dirname(os.path.realpath(__file__))
    vid_path = os.path.join(dir_path, "test.avi")
    # read video to varray
    varray = video2ndarray(vid_path,
            color_in="BGR",
            color_out="RGB")
    print(varray.shape)
    
    sums, nums = varray_sum_raw(varray)
    print(sums / nums)


if __name__ == "__main__":
    test_functions()