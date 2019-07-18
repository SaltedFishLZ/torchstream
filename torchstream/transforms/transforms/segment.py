__all__ = ["CenterSegment", "RandomSegment"]

from .. import functional as F


class CenterSegment(object):
    """
    Divide the entire video into several segments and sample the center part
    from each segment.
        |.........|.........|.........|.........|
             ^         ^         ^         ^
    Args:
        size (int): how many segments
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return F.center_segment(vid, s=self.size)

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)


class RandomSegment(object):
    """
    Divide the entire video into several segments and sample a random part
    from each segment.
        |.........|.........|.........|.........|
           ^             ^     ^              ^
    Args:
        size (int): how many segments
    """
    def __init__(self, size, bind=False):
        self.size = size
        self.bind = bind

    def __call__(self, vid):
        return F.random_segment(vid, s=self.size, bind=self.bind)

    def __repr__(self):
        string = self.__class__.__name__
        string += "(size={}, bind={})".format(self.size, self.bind)
        return string
