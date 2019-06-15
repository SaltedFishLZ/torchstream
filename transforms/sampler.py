import random
import numpy as np

import torchstream.transforms.functional as F
from torchstream.transforms.functional.blob import _is_varray

class FrameSampler(object):
    """
    """
    def __init__(self, size):
        """
        """
        self.size = size

    def __repr__(self):
        ret = self.__class__.__name__
        ret += " (size={})".format(self.size)

    def get_indices(self, vid):
        assert NotImplementedError("Abstract Base Shouldn't Be Used")
        return [0] * self.size

    def __call__(self, vid):
        """
        """
        assert _is_varray(vid), TypeError
        idx = self.get_indices(vid)
        return np.ascontiguousarray(vid[idx])


class RandomFrameSampler(FrameSampler):
    """
    Args:
        size (int): frame number
        shuffle (bool): whether return frames in-order
    """
    def __init__(self, size, shuffle=False):
        """
        """
        super(RandomFrameSampler, self).__init__(size)
        self.shuffle = shuffle

    def __repr__(self):
        ret = self.__class__.__name__
        ret += " (size={}, shuffle={})".format(self.size, self.shuffle)

    def get_indices(self, vid):
        t = vid.shape[0]
        idx = random.sample(range(t), self.size)    # sample without replacement
        if not self.shuffle:
            idx.sort()
        return idx
