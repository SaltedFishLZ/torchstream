"""
"""
import numpy as np
from .blob import _is_varray

def hflip(varray):
    """Horizontally flip the given varray.
    Args:
        varray (np.ndarray): video to be flipped.
    Returns:
        varray:  Horizontall flipped video.
    """
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))
    # We need to copy to make memory contiguous, otherwise ToTensor will crash
    # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    return np.flip(varray, 2).copy()

def vflip(varray):
    """Vertically flip the given varray.
    Args:
        varray (np.ndarray): video to be flipped.
    Returns:
        varray:  Vertically flipped video.
    """
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))
    return np.flip(varray, 1).copy()
