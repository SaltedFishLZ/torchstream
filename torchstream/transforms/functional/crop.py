import numbers
import numpy as np

from .typing import _is_varray


def crop(varray, i, j, h, w):
    """Crop the given varray.
    Args:
        varray (np.ndarray): video to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped video.
        w (int): Width of the cropped video.
    Returns:
        varray: Cropped video array.
    """
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))
    return varray[:, i : i + h, j : j + w, :]

def center_crop(varray, output_size):
    """
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = varray.shape[1:3]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(varray, i, j, th, tw)

