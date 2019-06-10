"""
"""
from __future__ import division
import random

from .. import functional as F


# -------------------------------- #
#           Video Flip             #
# -------------------------------- #
# NOTE: 
# 1. Flip might not apply to some datasets where
# motion information is important. 
# For example, some datasets need you to distinguish
# "From left to right" from "From right to left". If
# you use horizontal flip while not change the label,
# you will get an incorrect sample. But if you change
# the label, you may get very interesting augmentation.
# 2. For the flow modality, flipping the original video
# corresponds to get the opposite vector of the original
# flow, which is "flow = - flow", rather than "flow = 
# flip(flow)"


class RandomHorizontalFlip(object):
    """Horizontally flip the given video with a given probability.

    Args:
        p (float): probability of the image being flipped.
            Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        # TODO:
        # how to notify the caller whether the video is flipped ?
        if random.random() < self.p:
            return F.hflip(vid)
        return vid

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(object):
    """Vertically flip the given video with a given probability.

    Args:
        p (float): probability of the image being flipped.
            Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, vid):
        # TODO:
        # how to notify the caller whether the video is flipped ?
        if (random.random() < self.p):
            return F.vflip(vid)
        return vid

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
