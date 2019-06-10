"""
"""
import numpy as np

from .blob import _is_varray


def _get_snip_indices(t, s, mode):
    assert isinstance(t, int), TypeError
    assert isinstance(s, int), TypeError
    assert t > s, ValueError

    # interval (length of each segment) = floor(t/s)
    # it is the original way
    interval = t // s
    if interval <= 0:
        print("t={},s={}".format(t, s))
    offsets = []
    for i in range(s):
        offsets.append(i * interval)
    offsets = np.array(offsets)

    if mode == "center":
        indices = offsets + np.array([interval // 2] * s)
        indices = list(indices)
        return sorted(indices)
    elif mode == "random":
        indices = offsets + np.random.randint(low=0, high=interval, size=s)
        indices = list(indices)
        return sorted(indices)
    else:
        raise ValueError



def segment(vid, s, mode="center"):
    """Segment video in time dimension
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

    ## needs to be padded (zero padding)
    if t < s:
        pad_widths = ((0, s-t), (0, 0), (0, 0), (0, 0))
        constant_values=((0, 0), (0, 0), (0, 0), (0, 0))
        return np.pad(vid, pad_widths,
                      mode="constant",
                      constant_values=constant_values)

    ## NOTE: must use uint8
    vout = np.empty((s, h, w, c), dtype=np.uint8)
    snip_indices = _get_snip_indices(t, s, mode=mode)
    for idx in range(len(snip_indices)):
        vout[idx] = vid[snip_indices[idx]]

    return vout
