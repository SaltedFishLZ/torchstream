from __future__ import division
import random

from .. import functional as F

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vid):
        for t in self.transforms:
            vid = t(vid)
        return vid

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C) in the range [0, 255] to a
    torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0]
    if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, varray):
        """
        Args:
            varray (numpy.ndarray): varray to be converted to tensor.
        Returns:
            Tensor: Converted video.
        """
        return F.to_tensor(varray)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToVarray(object):
    """Convert a tensor to a varray
    CxTxHxW -> TxHxWxC
    """
    def __call__(self, vid):
        """
        """
        return F.to_varray(vid)

    def __repr__(self):
        return self.__class__.__name__ + '()'

