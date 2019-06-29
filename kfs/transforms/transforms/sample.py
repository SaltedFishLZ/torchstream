import random
import numpy as np

import torchstream.transforms.functional as F
from torchstream.transforms.functional.blob import _is_varray


class FrameSampler(object):
    """
    """
    def __init__(self, size, output_index=False, output_length=False):
        """
        """
        self.size = size
        self.output_index = output_index
        self.output_length = output_length

    def __repr__(self):
        ret = self.__class__.__name__
        ret += " (size={})".format(self.size)

    def get_indices(self, vid):
        """
        Args:
            return: a list/nparray
        """
        assert NotImplementedError("Abstract Base Shouldn't Be Used")
        return [0] * self.size

    def __call__(self, vid):
        """
        """
        assert _is_varray(vid), TypeError
        idx = self.get_indices(vid)
        
        if (not self.output_index) and (not self.output_length):
            return np.ascontiguousarray(vid[idx])
        else:
            ret = [np.ascontiguousarray(vid[idx]), ]
            if self.output_index:
                ret.append(idx)
            if self.output_length:
                ret.append([vid.shape[0]])
            return ret


class RandomFrameSampler(FrameSampler):
    """
    Args:
        size (int): frame number
        shuffle (bool): whether return frames in-order
    """
    def __init__(self, size, output_index=False, output_length=False, shuffle=False):
        """
        """
        super(RandomFrameSampler, self).__init__(size,
                                                 output_index=output_index,
                                                 output_length=output_length)
        self.shuffle = shuffle

    def __repr__(self):
        ret = self.__class__.__name__
        ret += " (size={}, shuffle={})".format(self.size, self.shuffle)

    def get_indices(self, vid):
        t = vid.shape[0]
        candidates = list(range(t))
        # padding
        while len(candidates) < self.size:
            candidates.append(candidates[-1])
        # sample without replacement
        idx = random.sample(candidates, self.size)
        if not self.shuffle:
            idx.sort()
        return idx
