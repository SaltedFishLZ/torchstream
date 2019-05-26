from __future__ import division
import sys
import math
import types
import random
import numbers
import warnings
import collections

import cv2
import torch
import numpy as np

from . import functional as F

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable



class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vid):
        for t in self.transforms:
            vid = t(vid)
        return vid

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C) in the range [0, 255] to a
    torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0]
    if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, varray):
        """
        Args:
            varray (numpy.ndarray): varray to be converted to tensor.
        Returns:
            Tensor: Converted video.
        """
        return F.to_tensor(varray)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToVarray(object):
    """Convert a tensor to a varray
    CxTxHxW -> TxHxWxC
    """
    def __call__(self, vid):
        """
        """
        return F.to_varray(vid)

    def __repr__(self):
        return self.__class__.__name__ + '()'



# -------------------------------- #
#        Video Normalize           #
# -------------------------------- #

class VideoNormalize(object):
    """Normalize a tensor video with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, 
    this transform will normalize each channel of the input ``torch.*Tensor``
    i.e. ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        paramstr = '(mean={0}, std={1})'.format(self.mean, self.std)
        return self.__class__.__name__ + paramstr


# -------------------------------- #
#          Video Padding           #
# -------------------------------- #

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


# -------------------------------- #
#           Video Crop             #
# -------------------------------- #

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
        w, h = vid.shape[1: 3]
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

class MultiScaleCrop(object):
    """Copy from TSM, need to be reconstructed later
    """
    def __init__(self, output_size,
                 scales=[1, .875, .75, .66],
                 max_distort=1, fix_crop=True,
                 more_fix_crop=True,
                 interpolation=cv2.INTER_LINEAR
                ):

        self.scales = scales
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.interpolation = interpolation
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, vid):

        img_size = vid.shape[1:2]

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(img_size)
        
        ## crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ## ret_img_group = [img.resize((self.output_size[0], self.output_size[1]), self.interpolation)
        ##                  for img in crop_img_group]
        ## return ret_img_group
        cropped_video = F.crop(offset_w, offset_h, crop_w, crop_h)
        resized_video = F.resize(cropped_video, self.output_size, self.interpolation)
        return resized_video

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
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
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret



# -------------------------------- #
#           Video Clip             #
# -------------------------------- #

class CenterClip(object):
    """
    """
    def __init__(self, size):
        self.size = size
    
    def __call__(self, vid):
        return F.center_clip(vid, self.size)        

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)

class RandomClip(object):
    """
    """
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(vid, output_size):
        t = vid.shape[0]
        tt = output_size
        if tt == t:
            return 0, t
        k = random.randint(0, t - tt)
        return k, tt

    def __call__(self, vid):
        k, tt = self.get_params(vid, self.size)
        return F.clip(vid, k, tt)

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)



# -------------------------------- #
#          Video Resize            #
# -------------------------------- #

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


