"""
"""
import random
import numpy as np

import sys
sys.path.append(".")
from torchstream.transforms.functional.blob import _is_varray


def center_segment(vid, s):
    """
    Args:
        s (int) sgement number
    """

    if not _is_varray(vid):
        raise TypeError('varray should be ndarray. Got {}'.format(vid))
    assert isinstance(s, int), TypeError

    t, h, w, c = vid.shape

    # short path
    if t == s:
        return vid

    # calculate indices
    interval = float(t) / float(s)
    indices = interval * np.array(range(s)) + interval / 2.0
    indices = np.uint(indices)
    indices.sort()
    indices = np.minimum(indices, t - 1)

    vout = vid[indices, :, :, :]
    return vout


def random_segment(vid, s, bind=False):
    """
    Args:
        s (int): sgement number
        bind (bool): fixed position in each segment
    """

    if not _is_varray(vid):
        raise TypeError('varray should be ndarray with dimension 4, data type uint8. Got {}'.format(vid))
    assert isinstance(s, int), TypeError

    t, h, w, c = vid.shape

    # short path
    if t == s:
        return vid

    # calculate indices
    interval = float(t) / float(s)
    offsets = interval * np.array(range(s))
    cursors = None
    if bind:
        cursors = random.uniform(0, interval)
    else:
        cursors = []
        while len(cursors) < s:
            cursors.append(random.uniform(0, interval))
        cursors = np.array(cursors)
    indices = offsets + cursors
    indices = np.uint(indices)
    indices.sort()
    indices = np.minimum(indices, t - 1)

    vout = vid[indices, :, :, :]
    return vout


if __name__ == "__main__":
    
    t = 100
    vid = np.empty(shape=(t, 224, 224, 3), dtype=np.uint8)

    vout = center_segment(vid, 101)
