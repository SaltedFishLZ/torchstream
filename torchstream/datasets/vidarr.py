"""VideoArray Wrapper
"""
__all__ = ["VideoArray"]

import numpy as np

from torchstream.datasets.utils.vision import video2ndarray
from torchstream.datasets.metadata.datapoint import DataPoint

class VideoArray(object):
    """
    A wrapper varray, supporting type-casting from Sample.
    """
    def __init__(self, x, **kwargs):
        """PyCharm will get wrong type checking
        """
        self.kwargs = kwargs
        if isinstance(x, DataPoint):
            assert not x.seq, NotImplementedError
            ## for the safety of your memory, we set "lazy" as
            #  the default mode
            self.lazy = kwargs.get("lazy", True)
            self.path = x.path
            self._array = None if self.lazy \
                else video2ndarray(x.path, **kwargs)
        else:
            self.lazy = False
            self.path = None
            self._array = np.array(x, **kwargs)

    def __repr__(self, idents=0):
        header = idents * "\t"
        string = header + "VideoArray\n"
        string += str(self._array)
        return string

    def get_varray(self):
        if self.lazy:
            _arr = video2ndarray(self.path, **self.kwargs)
        else:
            _arr = self._array
        assert len(_arr.shape) == 4, "Shape Error"
        return _arr

    def __array__(self):
        """Array casting
        Make this object array-like, which means it can be casted to
        np.ndarray.
        """
        return self.get_varray()


def _to_vidarr(x):
    """
    """
    assert isinstance(x, DataPoint), TypeError
    return VideoArray(x)




def test():

    import importlib

    dataset = "weizmann"
    metaset = importlib.import_module(
        "datasets.metadata.metasets.{}".format(dataset))

    kwargs = {
        "root" : metaset.AVI_DATA_PATH,
        "layout" : metaset.__layout__,
        "lbls" : metaset.__LABELS__,
        "mod" : "RGB",
        "ext" : "avi",
    }

    from .metadata.collect import collect_samples
    samples = collect_samples(**kwargs)

    for _sample in samples:
        vid_arr = VideoArray(_sample, lazy=False)
        print(vid_arr)
    # print(np.array(vid_arr))

if __name__ == "__main__":
    test()