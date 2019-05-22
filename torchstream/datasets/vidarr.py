"""VideoArray Wrapper
"""
__all__ = ["VideoArray"]

import numpy as np

from .utils.vision import video2ndarray
from .metadata.sample import Sample

class VideoArray(object):
    """
    A wrapper varray, supporting type-casting from Sample.
    """
    def __init__(self, x, **kwargs):
        """
        """
        if isinstance(x, Sample):
            ## parse args
            _kwargs = dict()
            if "cin" in kwargs:
                _kwargs["cin"] = kwargs["cin"]
            if "cout" in kwargs:
                _kwargs["cout"] = kwargs["cout"]
            ## get ndarray
            assert not x.seq, NotImplementedError
            self._array = video2ndarray(x.path, **_kwargs)
        else:
            self._array = np.array(x, **kwargs)
        assert len(self._array.shape) == 4, "Shape Error"

    def __repr__(self, idents=0):
        header = idents * "\t"
        string = header + "VideoArray\n"
        string += str(self._array)
        return string

    def __array__(self):
        """Array casting
        Make this object array-like, which means it can be casted to
        np.ndarray.
        """
        return self._array

    def get_varray(self):
        return self._array
        
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
        print(type(_sample))
        vid_arr = VideoArray(_sample)
        print(np.array(vid_arr))
    
if __name__ == "__main__":
    test()