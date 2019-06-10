from .. import functional as F


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
