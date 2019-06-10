"""
"""
from __future__ import division
import random

from .. import functional as F

class SpatialPad(object):
    r""" Spatial Padding
    """
    def __init__(self, padding, padding_mode='constant', **kwargs):
        self.padding = padding
        self.padding_mode = padding_mode
        self.kwargs = kwargs

    def __call__(self, img):
        """
        """
        return F.spad(img, self.padding, self.padding_mode, **(self.kwargs))

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, padding_mode={2})'.\
            format(self.padding, self.padding_mode)

class TemporalPad(object):
    r""" Temporal Padding
    """
    def __init__(self, padding, padding_mode='constant', **kwargs):
        self.padding = padding
        self.padding_mode = padding_mode
        self.kwargs = kwargs

    def __call__(self, img):
        """
        """
        return F.spad(img, self.padding, self.padding_mode, **(self.kwargs))

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, padding_mode={2})'.\
            format(self.padding, self.padding_mode)

