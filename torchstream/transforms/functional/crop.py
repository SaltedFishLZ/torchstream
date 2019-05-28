"""
"""
import numbers
import numpy as np
from .blob import _is_varray


def crop(vid, i, j, h, w):
    """Crop the given video.
    Args:
        vid (np.ndarray): video to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped video.
        w (int): Width of the cropped video.
    Returns:
        vid: Cropped video array.
    NOTE:
        The outside wrapper must gurantee that the indices are in range.
    """
    if not _is_varray(vid):
        raise TypeError('vid should be ndarray. Got {}'.format(vid))
    return vid[:, i : i + h, j : j + w, :]

def center_crop(vid, output_size):
    """Crop the given video in the center
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = vid.shape[1:3]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)


