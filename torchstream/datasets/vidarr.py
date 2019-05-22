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
    def __init__(self, x, lazy=True, **kwargs):
        """
        """
        if isinstance(x, Sample):
            ## parse args
            self.kwargs = dict()
            if "cin" in kwargs:
                self.kwargs["cin"] = kwargs["cin"]
            if "cout" in kwargs:
                self.kwargs["cout"] = kwargs["cout"]
            
            ## for the safety of your memory, we set "lazy" as 
            #  the default mode
            self.lazy = lazy

            assert not x.seq, NotImplementedError
            self.path = x.path
            if self.lazy:
                self._array = None
            else:
                self._array = video2ndarray(x.path, **self.kwargs)
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