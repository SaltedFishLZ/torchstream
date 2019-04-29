# -*- coding: utf-8 -*-
# Video Blob Transform
# Author: Zheng Liang
# 
# This module handles extra video data transformations which
# might not be included in the official PyTorch package.
import math
import copy
import time
import numbers

import numpy as np
import cv2
# import torch
# import torchvision

from .__init__ import *
from .video import *


__test__ = True


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
        h, w = varray.shape[1:3]
        th, tw = self.size
        # short cut
        if ((h == th) and (w == tw)):
            return(varray)
        # santity check
        assert (th <= h), "Crop height exceeds frame height"
        assert (tw <= w), "Crop width exceeds frame width"
        # generate offset
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        # crop
        result = varray[:, i : i + th, j : j + tw, :]
        return(result)

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)

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
        h, w = varray.shape[1:3]
        th, tw = self.size
        # short cut
        if ((h == th) and (w == tw)):
            return(varray)
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
        return self.__class__.__name__ + '(size={})'.format(self.size)



# -------------------------------- #
#          Video Resize            #
# -------------------------------- #

class VideoResize(object):
    '''
    Resize a video via OpenCV's resize API
    NOTE: Currently, we only support spatial resize.
    '''
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        '''
        - size 
        - interpolation
        '''
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = copy.deepcopy(size)
        self.interpolation = interpolation
    
    def __call__(self, varray):
        '''
        - varray : [T][H][W][C]
        - return : [T][H][w][C]
        '''
        t, h, w, c = varray.shape
        result_shape = (t, self.size[0], self.size[1], c)
        result = np.empty(result_shape, np.dtype('float32'))
        for _i in range(t):
            farray = varray[_i, :, :, :]
            result[_i, :, :, :] = cv2.resize(farray, 
                dsize=(self.size[1], self.size[0]),
                interpolation=self.interpolation)
        return(result)

    def __repr__(self):
        return self.__class__.__name__ + '(size={}, intrpl={})'.\
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
class VideoFlip(object):
    '''
    Video level deterministic flip
    '''
    def __init__(self, dim="H"):
        '''
        Initialization
        - dim : "H", "W" or "T", which dimension the flip
        takes place. "H" is height, "W" is width, "T" is time.
        '''
        assert (dim in ["H", "W", "T"]), "Unsupported flip dimension"
        self.dim = copy.deepcopy(dim)
    
    def __call__(self, varray):
        '''
        Flip a video, must execute
        - varray : [T][H][W][C]
        - return : flipped video
        '''
        if ("T" == self.dim):
            return(np.flip(varray, 0))
        elif ("H" == self.dim):
            return(np.flip(varray, 1))
        elif ("W" == self.dim):
            return(np.flip(varray, 2))
        else:
            assert True, "Error in transform"

    def __repr__(self):
        return self.__class__.__name__ + '(dim={})'.format(self.dim)


class VideoRandomFlip(VideoFlip):
    '''
    Video level random flip
    '''
    def __init__(self, dim="H"):
        super(VideoRandomFlip, self).__init__(dim=dim)
    
    def __call__(self, varray):
        # TODO: how to notify the caller whether the video is flipped ?
        '''
        Flip a video, execute with a probability of 0.5
        - varray : [T][H][W][C]
        - return : flipped video
        '''
        v = np.random.random()
        # identical
        if (v < 0.5):
            return(varray)
        # otherwise, flip
        if ("T" == self.dim):
            return(np.flip(varray, 0))
        elif ("H" == self.dim):
            return(np.flip(varray, 1))
        elif ("W" == self.dim):
            return(np.flip(varray, 2))
        else:
            assert True, "Error in transform"




# -------------------------------- #
#        Video Normalize           #
# -------------------------------- #

class VideoNormalize(object):
    '''
    Normalize all pixels of a video for each channel
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

    def __repr__(self):
        return self.__class__.__name__ + '(means={}, stds={})'.\
            format(self.means, self.stds)


# -------------------------------- #
#       PyTorch Tensor API         #
# -------------------------------- #

class ToTensor(object):
    '''
    Convert a video sequence ndarray which is stored as [T][H][W][C] to 
    PyTorch float tensor [T][C][H][W].
    NOTE: Orginal video pixel values are np.uint8, in [0, 255], if you want
    to scale the value to [0, 1], please specify the 'scale' argument in 
    __init__ with True. If use normalization before, there is no need to scale.
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





# -------------------------------- #
#     Self Testing Utillities      #
# -------------------------------- #

def test_transforms(test_configuration):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    vid_path = os.path.join(dir_path, "test.avi")
    # read video to varray
    varray = video2ndarray(vid_path,
            color_in=test_configuration['video_color'],
            color_out=test_configuration['varray_color'])
    # # test video crop
    # crop = VideoRandomCrop(size=(128,171))
    # varray = crop(varray)
    # print(varray.shape)
    # _f = varray.shape[0]
    # for _i in range(_f):
    #     winname = "{}".format(_i)
    #     farray_show(winname, varray[_i,:,:,:])
    #     cv2.moveWindow(winname, 40,30) 
    #     (cv2.waitKey(0) & 0xFF == ord('q'))
    #     cv2.destroyAllWindows()

    # # test video flip
    # flip = VideoRandomFlip(dim="W")
    # varray = flip(varray)
    # print(varray.shape)
    # _f = varray.shape[0]
    # for _i in range(_f):
    #     winname = "{}".format(_i)
    #     farray_show(winname, varray[_i,:,:,:])
    #     cv2.moveWindow(winname, 40,30) 
    #     (cv2.waitKey(0) & 0xFF == ord('q'))
    #     cv2.destroyAllWindows()

    # test video normalization
    resize = VideoResize(size=(720, 720))
    varray = resize(varray)
    print(varray.shape)
    _f = varray.shape[0]
    for _i in range(_f):
        winname = "{}".format(_i)
        farray_show(winname, varray[_i,:,:,:])
        cv2.moveWindow(winname, 40,30) 
        (cv2.waitKey(0) & 0xFF == ord('q'))
        cv2.destroyAllWindows()


if __name__ == "__main__":


    if (__test__):

        test_configuration = {
            'video_color'   : "BGR",
            'varray_color'  : "RGB",
            'frames_color'  : "BGR",
            'imgseq_color'  : "RGB"
        }

        test_transforms(test_configuration)