# -*- coding: utf-8 -*-
# Video Blob Transform
# Author: Zheng Liang
# 
# This module handles extra video data transformations which
# might not be included in the official PyTorch package.
import math
import copy
import random
import numbers

import numpy as np
import cv2
import torch
import torchvision

# -------------------------------- #
#           Video Crop             #
# -------------------------------- #

class VideoRandomCrop(object):
    '''
    Random crop an certain area in the [H][W] dimension and
    keep all frames in a crop consistent in [H][W]
    '''
    def __init__(self, size):
        '''
        Initialization function
        - size : an integer S or a tuple (H, W)
        '''
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, varray):
        '''
        Video level random crop
        - varray : video in a numpy ndarray, data layout is [T][H][W][C]
        - return : a cropped varray with the same data layout
        '''
        h, w = varray.size
        th, tw = self.size
        # santity check
        assert (th <= h), "Crop height exceeds frame height"
        assert (tw <= w), "Crop width exceeds frame width"
        # generate offset
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        # crop
        result = varray[:, i : i + th, j : j + tw, :]
        return(result)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class VideoCenterCrop(object):
    '''
    Crop the center area in the [H][W] dimension and
    keep all frames in a crop consistent in [H][W]    
    '''
    def __init__(self, size):
        '''
        Initialization function
        - size : an integer S or a tuple (H, W)
        '''
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, varray):
        '''
        Video level random crop
        - varray : video in a numpy ndarray, data layout is [T][H][W][C]
        - return : a cropped varray with the same data layout
        '''
        h, w = varray.size
        th, tw = self.size
        # santity check
        assert (th <= h), "Crop height exceeds frame height"
        assert (tw <= w), "Crop width exceeds frame width"
        # generate offset
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        # crop
        result = varray[:, i : i + th, j : j + tw, :]
        return(result)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)



# -------------------------------- #
#           Video Flip             #
# -------------------------------- #
# NOTE: 
# Flip might not apply to some datasets where
# motion information is important. 
# For example, some datasets need you to distinguish
# "From left to right" from "From right to left". If
# you use horizontal flip while not change the label,
# you will get an incorrect sample.



# -------------------------------- #
#        Video Normalize           #
# -------------------------------- #

class VideoNormalize(object):
    '''
    '''
    def __init__(self, means, stds):
        '''
        Initialization
        - means : pixel mean values for all pixels ([T][H][W]) in
        different channels
        - stds : pixel standard deviations for all pixels in different
        channels
        '''
        self.means = copy.deepcopy(means)
        self.stds = copy.deepcopy(stds)

    def __call__(self, varray):
        '''
        Normalize a video for each channel
        - varray : input video as a Numpy ndarray in [T][H][W][C] format
        - return : a normalized varray with the same format as input
        '''
        (_t, _h, _w) = varray.shape[0:3]
        result = varray - np.tile(self.means, (_t, _h, _w, 1))
        result = varray / np.tile(self.stds, (_t, _h, _w, 1))
        return(result)


# -------------------------------- #
#          Video Scale             #
# -------------------------------- #


# -------------------------------- #
#       PyTorch Tensor API         #
# -------------------------------- #

class ToTensor(object):
    '''
    Convert a video sequence ndarray which is stored as [T][H][W][C] to 
    PyTorch float tensor [T][C][H][W].
    NOTE: Orginal video pixel values are np.uint8, in [0, 255], if you want
    to scale the value to [0, 1], please use normalization before, or specify
    the 'scale' argument in __init__ as True.
    '''
    def __init__(self, val_scale=False):
        '''
        Initialization function
        - scale : whether the input blob will be scaled from [0,255] to [0,1]
        '''
        self.val_scale = val_scale

    def __call__(self, varray):
        '''
        Transform a varray to a PyTorch tensor, while make sure the data 
        layout is right.
        - varray : input video array as a Numpy ndarray, [T][H][W][C]
        '''    
        ret = torch.from_numpy(varray).permute(3, 2, 0, 1).contiguous()
        if (self.val_scale):
            ret = ret.div(255.0)
        return ret


if __name__ == "__main__":
    pass