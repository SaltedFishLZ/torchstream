"""
"""
import numpy as np
from .typing import _is_varray

def hflip(varray):
    """Horizontally flip the given varray.
    Args:
        varray (np.ndarray): video to be flipped.
    Returns:
        varray:  Horizontall flipped video.
    """
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))
    return np.flip(varray, 2)

def vflip(varray):
    """Vertically flip the given varray.
    Args:
        varray (np.ndarray): video to be flipped.
    Returns:
        varray:  Vertically flipped video.
    """
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))
    return np.flip(varray, 1)
