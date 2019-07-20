""" Clipping for videos
TODO Supporting Tensor
"""
from .blob import _is_varray


def clip(varray, k, t):
    """Get [k, k+t) part in the time dimension for a video of duration T
    """
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))
    return varray[k: k + t, :, :, :]


def center_clip(vid, output_size):
    """Clip the given video in the middle
    """
    t = vid.shape[0]
    tt = output_size
    k = int(round((t - tt) / 2.))
    return clip(vid, k, tt)
