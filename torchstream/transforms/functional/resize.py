
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
    _shape = (t, size[0], size[1], c)
    
    result = np.empty(_shape, np.dtype("float32"))
    for _i in range(t):
        farray = varray[_i, :, :, :]
        farray = cv2.resize(farray, dsize=(size[1], size[0]),
                            interpolation=interpolation)
        result[_i, :, :, :] = farray
    return result
