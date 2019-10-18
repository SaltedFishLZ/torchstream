"""
"""
import numbers
import cv2
import sys
import collections

from .. import functional as F


if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class Resize(object):
    """
    Resize a video via OpenCV"s resize API
    NOTE: Currently, we only support spatial resize.
    """
    def __init__(self, size, threshold=None,
                 interpolation=cv2.INTER_LINEAR):
        """
        Args:
            size (int): target size
            threshold (int/tuple): input
            interpolation (int)
        Return:
            varray
        """
        if isinstance(size, numbers.Number):
            self.size = int(size)
        elif (isinstance(size, Iterable) and len(size) == 2):
            self.size = tuple(size)
        else:
            raise TypeError

        if threshold is None:
            self.threshold = threshold
        elif isinstance(threshold, numbers.Number):
            self.threshold = int(threshold)
        else:
            raise TypeError

        self.interpolation = interpolation

    def __repr__(self):
        return self.__class__.__name__ + "(size={}, interpolation={})".\
            format(self.size, self.interpolation)

    def __call__(self, vid):
        _, h, w, _ = vid.shape
        if (self.threshold is None) or (min(h, w) >= self.threshold):
            return F.resize(vid, self.size, self.interpolation)
        return vid
