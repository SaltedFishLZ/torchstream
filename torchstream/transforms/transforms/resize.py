"""
"""
import numbers
import cv2

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
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, vid):
        """
        """
        return F.resize(vid, self.size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + "(size={}, interpolation={})".\
            format(self.size, self.interpolation)


