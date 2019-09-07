"""Preprocess datasets
"""
import copy

from torchstream.io.conversion import vid2vid
from torchstream.io.datapoint import DataPoint
from torchstream.utils.mapreduce import Manager


def dataset_vid2vid(name, datapoints, dst_root, dst_ext="avi",
                    worker_num=16, **kwargs):
    """transform video format
    """
    manager = Manager(name="converting dataset [{}]".format(name),
                      mapper=vid2vid,
                      retries=10,
                      **kwargs)
    manager.hire(worker_num=worker_num)
    tasks = []

    for src in datapoints:
        assert isinstance(src, DataPoint), TypeError
        dst = copy.deepcopy(src)
        dst.root = dst_root
        dst.ext = dst_ext
        dst._path = dst.path
        tasks.append({"src": src, "dste": dst})

    successes = manager.launch(tasks=tasks, enable_tqdm=True)
    print(successes)


if __name__ == "__main__":
    pass
    # TODO:
    # collecting datapoints
