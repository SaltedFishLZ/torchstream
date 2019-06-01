"""
"""
import random
import numbers
import numpy as np

from torchstream.transforms.functional import crop, center_crop, resize


def upper_left_crop(vid, output_size):
    """Crop the given video in the upper left
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    th, tw = output_size
    i = 0
    j = 0
    return crop(vid, i, j, th, tw)



def upper_center_crop(vid, output_size):
    """Crop the given video in the upper center
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = vid.shape[1:3]
    th, tw = output_size
    i = 0
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)



def upper_right_crop(vid, output_size):
    """Crop the given video in the upper right
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = vid.shape[1:3]
    th, tw = output_size
    i = 0
    j = w - tw
    return crop(vid, i, j, th, tw)





def center_left_crop(vid, output_size):
    """Crop the given video in the center left
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = vid.shape[1:3]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = 0
    return crop(vid, i, j, th, tw)


def center_right_crop(vid, output_size):
    """Crop the given video in the center right
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = vid.shape[1:3]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = w - tw
    return crop(vid, i, j, th, tw)





def lower_left_crop(vid, output_size):
    """Crop the given video in the lower left
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = vid.shape[1:3]
    th, tw = output_size
    i = h - th
    j = 0
    return crop(vid, i, j, th, tw)






def lower_center_crop(vid, output_size):
    """Crop the given video in the lower center
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = vid.shape[1:3]
    th, tw = output_size
    i = h - th
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)





def lower_right_crop(vid, output_size):
    """Crop the given video in the lower right
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    h, w = vid.shape[1:3]
    th, tw = output_size
    i = h - th
    j = w - tw
    return crop(vid, i, j, th, tw)





def five_crop(vid, output_size):
    """
    Args:
    Returns:
        tuple: a tuple of 5 crops
    """
    ul = upper_left_crop(vid, output_size)
    ur = upper_right_crop(vid, output_size)
    ll = lower_left_crop(vid, output_size)
    lr = lower_right_crop(vid, output_size)
    center = center_crop(vid, output_size)
    return (ul, ur, ll, lr, center)

def one_of_five_crop(vid, output_size):
    """
    """
    transforms = [
        upper_left_crop,
        upper_right_crop,
        lower_left_crop,
        lower_right_crop,
        center_crop
    ]
    transform = random.choice(transforms)
    return transform(vid, output_size)


def nine_crop(vid, output_size):
    """
    """
    ul = upper_left_crop(vid, output_size)
    uc = upper_center_crop(vid, output_size)
    ur = upper_right_crop(vid, output_size)

    cl = center_left_crop(vid, output_size)
    cc = center_crop(vid, output_size)
    cr = center_right_crop(vid, output_size)

    ll = lower_left_crop(vid, output_size)
    lc = lower_center_crop(vid, output_size)
    lr = lower_right_crop(vid, output_size)

    return(
        (
            ul, uc, ur,
            cl, cc, cr,
            ll, lc, lr
        )
    )

def one_of_nine_crop(vid, output_size):
    """
    """
    transforms = [
        upper_left_crop,
        upper_center_crop,
        upper_right_crop,
        center_left_crop,
        center_crop,
        center_right_crop,
        lower_left_crop,
        lower_center_crop,
        lower_right_crop,
    ]
    transform = random.choice(transforms)
    return transform(vid, output_size)