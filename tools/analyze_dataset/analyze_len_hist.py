"""
"""
import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt


from torchstream.datasets import analysis
from torchstream.datasets.utils.mapreduce import Manager
from torchstream.datasets.metadata.collect import collect_datapoints

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(__file__)
ANALY_PATH = os.path.join(DIR_PATH, ".analyzed.d")


def len_hist(name, samples, worker_num=80, bins=20, **kwargs):
    """
    """
    def filter_len(results):
        ret = []
        for _result in results:
            if _result < 750:
                ret.append(_result)
        return ret

    manager = Manager(name="Get Length Hist [{}]".format(name),
                      mapper=analysis.sample_len,
                      reducer=filter_len,
                      retries=10,
                      max=500,
                      **kwargs
                      )
    manager.hire(worker_num=worker_num)

    print("Assembleing Tasks")
    tasks = []
    for _sample in samples:
        tasks.append({"sample": _sample})

    print("Lanuching Jobs")
    lens = manager.launch(tasks=tasks, enable_tqdm=True)

    print("Min Len", min(lens))
    print("Max Len", max(lens))

    nphist = np.histogram(lens, bins=bins)
    print("Numpy Hist")
    print("Density", nphist[0])
    print("Bins", nphist[1])

    plt.hist(lens, density=True, bins=bins)
    plt.savefig(os.path.join(ANALY_PATH, name + ".len.dist.density.pdf"),
                bbox_inches="tight")


def main(name):

    import importlib
    metaset = "torchstream.datasets.metadata.metasets.{}".format(name)
    metaset = importlib.import_module(metaset)

    kwargs = {
        "root" : metaset.JPG_DATA_PATH,
        "layout" : metaset.__layout__,
        "lbls" : metaset.__LABELS__,
        "mod" : "RGB",
        "ext" : "jpg",
    }

    if hasattr(metaset, "AVI_DATA_PATH"):
        if os.path.exists(metaset.AVI_DATA_PATH):
            kwargs["root"] = metaset.AVI_DATA_PATH
            kwargs["ext"] = "avi"
        else:
            print("Warning: Failed to find AVI dataset")

    if hasattr(metaset, "__ANNOTATIONS__"):
        kwargs["annots"] = metaset.__ANNOTATIONS__

    if hasattr(metaset, "JPG_FILE_TMPL"):
        kwargs["tmpl"] = metaset.JPG_FILE_TMPL

    if hasattr(metaset, "JPG_IDX_OFFSET"):
        kwargs["offset"] = metaset.JPG_IDX_OFFSET

    print("Collecting Datapoints")
    samples = collect_datapoints(**kwargs)

    len_hist(name, samples, **kwargs)

if __name__ == "__main__":
    print(sys.argv)
    for _i in range(1, len(sys.argv)):
        main(sys.argv[_i])
