"""
"""
import numpy as np
from .blob import _is_varray


def hflip(varray):
    """Horizontally flip the given varray.
    Note:
        We need to copy to make memory contiguous, you can refer to:
        https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    Args:
        varray (np.ndarray): video to be flipped.
    Returns:
        varray:  Horizontall flipped video.
    """  # noqa
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))
    return np.flip(varray, 2).copy()


def vflip(varray):
    """Vertically flip the given varray.
    Note:
        We need to copy to make memory contiguous, you can refer to:
        https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    Args:
        varray (np.ndarray): video to be flipped.
    Returns:
        varray:  Vertically flipped video.
    """
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))
    return np.flip(varray, 1).copy()
