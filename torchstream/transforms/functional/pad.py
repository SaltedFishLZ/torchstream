import sys
import numbers
import collections

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

import numpy as np

from .blob import _is_varray

SUPPORTED_PADDING_MODES = ["constant", "edge", "reflect", "symmetric"]

def pad(vid, padding, padding_mode="constant", **kwargs):
    r"""Pad, pad the given video in the [T][H][W] dimensions

    Args:
        vid (varray): Video to be padded.

        padding (tuple): Padding on each border.
    Returns:
        Varray: Padded video.
    """
    assert _is_varray(vid), TypeError
    if not isinstance(padding, tuple):
        raise TypeError('Got inappropriate padding arg')
    if len(padding) != 6:
        raise ValueError
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    assert padding_mode in SUPPORTED_PADDING_MODES, TypeError

    pad_left = padding[0]
    pad_top = padding[1]
    pad_right = padding[2]
    pad_bottom = padding[3]
    pad_before = padding[4]
    pad_after = padding[5]

    pad_shape = ((pad_before, pad_after), (pad_top, pad_bottom), 
                 (pad_left, pad_right), (0, 0))

    return np.pad(vid, pad_shape, padding_mode)


def spad(vid, padding, padding_mode="constant", **kwargs):
    r"""Spatial Pad, pad the given video on the [H][W] dimension
    """
    assert _is_varray(vid), TypeError
    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    assert padding_mode in SUPPORTED_PADDING_MODES, TypeError

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    padded_vid = np.pad(vid, (pad_left, pad_top, pad_right, pad_bottom, 0, 0), padding_mode)
    return padded_vid


def tpad(vid, padding, padding_mode="constant", **kwargs):
    r"""Temporal Pad, pad the given video on the [T] dimension
    """
    assert _is_varray(vid), TypeError
    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')
    assert padding_mode in SUPPORTED_PADDING_MODES, TypeError

    if isinstance(padding, int):
        pad_before = pad_after = padding
    if isinstance(padding, Sequence) and len(padding) == 2:
        pad_before = padding[0]
        pad_after = padding[1]

    padded_vid = np.pad(vid, (0, 0, 0, 0, pad_before, pad_after), padding_mode)
    return padded_vid
