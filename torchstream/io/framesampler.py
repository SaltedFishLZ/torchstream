"""Frame sampler for image sequence
"""
__all__ = ["CenterSegmentFrameSampler", "RandomSegmentFrameSampler"]
import copy

from .datapoint import DataPoint
import torchstream.transforms.functional.segment as segment


class CenterSegmentFrameSampler(object):
    """
    Args:
        size (int) number of segments
    """
    def __init__(self, size):
        self.size = size

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)

    def __call__(self, framepaths):
        """
        Args:
            framepaths
        """
        t = len(framepaths)
        indices = segment.center_segment_indices(t, self.size)
        results = [framepaths[idx] for idx in indices]

        return results


class RandomSegmentFrameSampler(object):
    """
    Args:
        size (int) number of segments
    """
    def __init__(self, size, bind=False):
        self.size = size
        self.bind = bind

    def __repr__(self):
        string = self.__class__.__name__
        string += "(size={}, bind={})".format(self.size, self.bind)
        return string

    def __call__(self, framepaths):
        """
        Args:
            framepaths
        """
        t = len(framepaths)
        indices = segment.random_segment_indices(t,
                                                 self.size,
                                                 self.bind)
        results = [framepaths[idx] for idx in indices]

        return results
