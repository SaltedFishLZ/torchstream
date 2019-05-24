""" Clipping for videos 
TODO Supporting Tensor
"""
from .typing import _is_varray

def clip(varray, t, l):
    """Get [t, t+l) part for a [T][H][W][C] video
    """
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))
    return varray[t : t + l, :, :, :]