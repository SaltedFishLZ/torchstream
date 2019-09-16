import sys
import collections
if sys.version_info < (3, 3):
    Iterable = collections.Iterable
else:
    Iterable = collections.abc.Iterable

import cv2
import numpy as np

from .blob import _is_varray

def resize(varray, size, interpolation=cv2.INTER_LINEAR):
    """resize a video via OpenCV"s resize API
    NOTE: Currently, we only support spatial resize.
    """
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))
    t, h, w, c = varray.shape

    oh = None
    ow = None
    if isinstance(size, int):
        # short path
        if (w <= h and w == size) or (h <= w and h == size):
            return varray 
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    elif isinstance(size, Iterable) and (len(size) == 2):
        oh, ow = tuple(size)
    else:
        raise TypeError("Invalid size {}".format(size))

    _shape = (t, oh, ow, c)
    result = np.empty(_shape, dtype=np.uint8)
    for _i in range(t):
        farray = varray[_i, :, :, :]
        farray = cv2.resize(farray, dsize=(ow, oh),
                            interpolation=interpolation)
        result[_i, :, :, :] = farray
    return result
