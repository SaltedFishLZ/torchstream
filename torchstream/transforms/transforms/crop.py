"""
"""
from __future__ import division
import random
import numbers

import cv2

from .. import functional as F


class CenterCrop(object):
    """ Crop a given video spatially in the center area
    Crop in the [H][W] dimension and keep all frames consistent
    in crop location.

    Args
        size : an integer S or a tuple (H, W)
    """
    def __init__(self, size):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, varray):
        return F.center_crop(varray, self.size)

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)

class RandomCrop(object):
    """Crop the given video at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(vid, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            vid (varray): video to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop``.
        """
        h, w = vid.shape[1: 3]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.size)
        return F.crop(vid, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)

class FiveCrop(object):
    """
    Args:
        size : an integer S or a tuple (H, W)
    Returns:
        tuple: a tuple of 5 crops
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, vid):
        return F.five_crop(vid, self.size)

class NineCrop(object):
    """
    Args:
        size : an integer S or a tuple (H, W)
    Returns:
        tuple: a tuple of 9 crops
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, vid):
        return F.nine_crop(vid, self.size)

class MultiScaleCrop(object):
    """Randomly apply 1 of multiple scales crop
    """
    def __init__(self, output_size,
                 scales=[1, 0.875, 0.75, 0.66],
                 max_distort=1,
                 more_fix_crop=True,
                 interpolation=cv2.INTER_LINEAR,
                 **kwargs
                ):

        self.scales = scales
        self.max_distort = max_distort
        self.more_fix_crop = more_fix_crop
        self.interpolation = interpolation
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, vid):

        output_size = self._sample_crop_size(vid)
        
        cropped_video = F.one_of_nine_crop(vid, output_size)

        resized_video = F.resize(cropped_video, self.output_size, self.interpolation)
        
        return resized_video


    def _sample_crop_size(self, vid):
        h, w = vid.shape[1:3]
        
        ## generate a set of crop sizes
        base_size = min(h, w)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.output_size[1] if abs(x - self.output_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.output_size[0] if abs(x - self.output_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
        
        ## random sample 1 size
        crop_pair = random.choice(pairs)

        return crop_pair
