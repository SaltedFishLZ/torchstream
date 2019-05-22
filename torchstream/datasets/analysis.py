"""Dataset Analysis Toolbox
"""
import numpy as np

from .vidarr import VideoArray
from .imgseq import ImageSequence
from .metadata.sample import Sample

# ---------------------------------------------------------------- #
#                       Pixel Level Statistics                     #
# ---------------------------------------------------------------- #

## Get the sum of all pixels in a varray for each channel
#  
#  @param varray np.ndarray: input blob
#  @param return tuple: (sum array, pixel number array), we use array for 
#  multi-channels.
def varray_sum(varray, **kwargs):
    """
    Get the sum of all pixels in a varray for each channel
    """
    ## get frame shape, leave the channel dimension alone
    (_t, _h, _w) = varray.shape[0:3]
    nums = _t * _h * _w
    ## Numpy sum over multiple axes
    #  refenrec: https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    sums = varray.sum(axis=(0, 1, 2))
    return(sums, nums)

## Get the RSS(residual sum of squares) of all pixels in a varray of 
#  each channel.
#  
#  @param varray np.ndarray: input blob
#  @param means np.ndarray: means of the varray, if not specified, this function will
#  call varray_sum_raw to calculate it.
#  @param return return tuple: (rss array, pixel numbers), we use array for 
#  multi-channels.
#  Reference:
#  https://en.wikipedia.org/wiki/Residual_sum_of_squares
def varray_rss(varray, means=None, **kwargs):
    """
    """
    if means is None:
        sums, nums = varray_sum(varray)
        means = sums / nums

    (_t, _h, _w) = varray.shape[0:3]
    nums = _t * _h * _w
    residuals = varray - np.tile(means, (_t, _h, _w, 1))
    residual_squares = np.square(residuals)

    ## Numpy sum over multiple axes
    #  https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    rsses = residual_squares.sum(axis=(0, 1, 2))

    return(rsses, nums)

def sample_sum(sample, **kwargs):
    """Wrapper
    """
    assert isinstance(sample, Sample), TypeError
    if sample.seq:
        varray = np.array(ImageSequence(sample, **kwargs))
    else:
        varray = np.array(VideoArray(sample, **kwargs))
    return varray_sum(varray, **kwargs)

def sample_rss(sample, **kwargs):
    """Wrapper
    """
    assert isinstance(sample, Sample), TypeError
    if sample.seq:
        varray = np.array(ImageSequence(sample, **kwargs))
    else:
        varray = np.array(VideoArray(sample, **kwargs))
    return varray_sum(varray, **kwargs)


# ---------------------------------------------------------------- #
#                       Frame Level Statistics                     #
# ---------------------------------------------------------------- #
                                                                    
def varray_len(varray, **kwargs):
    """Duration of a varray
    """
    return varray.shape[0]

def varray_hxw(varray, **kwargs):
    """ (H, W) of a varray
    """
    return varray.shape[1:3]

def sample_len(sample, **kwargs):
    """
    """
    assert isinstance(sample, Sample), TypeError
    ## short path for image sequence
    if sample.seq:
        img_seq = ImageSequence(sample, **kwargs)
        return img_seq.fcount
    ## call varray_len
    varray = np.array(VideoArray(sample, **kwargs))
    return varray_len(varray, **kwargs)

def sample_hxw(sample, **kwargs):
    """
    """
    assert isinstance(sample, Sample), TypeError
    ## short path for image sequence
    if sample.seq:
        img_seq = ImageSequence(sample, **kwargs)
        farray = img_seq.get_farray(idx=0)
        return farray.shape[0:2]
    ## call varray_hxw
    varray = np.array(VideoArray(sample, **kwargs))
    return varray_hxw(varray, **kwargs)

