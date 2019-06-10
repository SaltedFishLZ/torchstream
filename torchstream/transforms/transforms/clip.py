"""
"""
import random
from .. import functional as F

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
