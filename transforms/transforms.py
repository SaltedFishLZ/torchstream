import numbers
import random

import cv2

from torchstream.transforms.transform import RandomCrop


from . import functional as F



class IdentityTransform(object):
    """IdentityTransform
    """
    def __call__(self, data):
        return data
    def __repr__(self):
        return self.__class__.__name__



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







class CenterSegment(object):
    """
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return F.segment(vid, s=self.size, mode="center")

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)


class RandomSegment(object):
    """
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return F.segment(vid, s=self.size, mode="random")

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)







class RandomTransforms(object):
    """Base class for a list of transformations with randomness
    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string



class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability
    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, vid):
        if self.p < random.random():
            return vid
        for t in self.transforms:
            vid = t(vid)
        return vid

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order
    """
    def __call__(self, vid):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            vid = self.transforms[i](vid)
        return vid


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """
    def __call__(self, vid):
        t = random.choice(self.transforms)
        return t(vid)



class MultiScaleCrop(object):
    """Randomly apply 1 of multiple scales crop
    """
    def __init__(self, output_size,
                 scales=[1, .875, .75, .66],
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



