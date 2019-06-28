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

def fps_hist(name, samples, worker_num=80, **kwargs):
    """
    """

    manager = Manager(name="Get FPS Hist [{}]".format(name),
                      mapper=analysis.sample_fps,
                      # retries=10,
                      # **kwargs
                      )
    manager.hire(worker_num=worker_num)

    print("Assembleing Tasks")
    tasks = []
    for _sample in samples:
        tasks.append({"sample": _sample})

    print("Lanuching Jobs")
    fpses = manager.launch(tasks=tasks, enable_tqdm=True)

    for _fps in fpses:
        if isinstance(_fps, int):
            print("[{}]".format(_fps))
    print("Min FPS", min(fpses))
    print("Max FPS", max(fpses))

    nphist = np.histogram(fpses, bins=20)
    print("Numpy Hist")
    print("Density", nphist[0])
    print("Bins", nphist[1])

    plt.hist(fpses, density=True, bins=20)

    os.makedirs(ANALY_PATH, exist_ok=True)
    plt.savefig(os.path.join(ANALY_PATH, name + ".fps.dist.density.pdf"),
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

    fps_hist(name, samples, **kwargs)

if __name__ == "__main__":
    print(sys.argv)
    for _i in range(1, len(sys.argv)):
        main(sys.argv[_i])
