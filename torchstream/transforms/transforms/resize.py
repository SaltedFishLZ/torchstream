"""
"""
import numbers
import cv2
import sys
import collections
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

from .. import functional as F


class Resize(object):
    """
    Resize a video via OpenCV"s resize API
    NOTE: Currently, we only support spatial resize.
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        """
        - size 
        - interpolation
        """
        if isinstance(size, numbers.Number):
            self.size = int(size)
        elif (isinstance(size, Iterable) and len(size) == 2):
            self.size = tuple(size)
        else:
            raise TypeError
        self.interpolation = interpolation

    def __call__(self, vid):
        """
        """
        return F.resize(vid, self.size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + "(size={}, interpolation={})".\
            format(self.size, self.interpolation)


