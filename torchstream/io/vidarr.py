"""VideoArray Wrapper
"""
__all__ = ["VideoArray"]

import numpy as np

from .datapoint import DataPoint
from .backends.opencv import video2ndarray


class VideoArray(object):
    """
    A wrapper for a video file, will return a varray.
    Currently, only support RGB/BGR input.
    """
    def __init__(self, datapoint, lazy=True, volatile=True, **kwargs):
        """
        NOTE: PyCharm will get wrong type checking (isinstance)
        """
        assert isinstance(datapoint, DataPoint), TypeError
        assert not datapoint.seq, TypeError

        self.datapoint = datapoint
        self.kwargs = kwargs

        # for the safety of your memory, we set "lazy" & "volatile"
        # as the default mode
        self.lazy = lazy
        self.volatile = volatile

        self._array = None
        if not self.lazy
            self._array = video2ndarray(self.path, **kwargs)

    @property
    def path(self):
        return self.datapoint.path

    def __repr__(self, idents=0):
        header = idents * "\t"
        string = header + "VideoArray\n"
        string += str(self._array)
        return string

    def get_varray(self):
        if self.lazy:
            if self.volatile:
                _arr = video2ndarray(self.path, **self.kwargs)
            else:
                if self._array is None:
                    self._array = video2ndarray(self.path, **self.kwargs)
                _arr = self._array
        else:
            _arr = self._array
        return _arr

    def __array__(self):
        """Array casting
        Make this object array-like, which means it can be casted to
        np.ndarray.
        """
        return self.get_varray()
