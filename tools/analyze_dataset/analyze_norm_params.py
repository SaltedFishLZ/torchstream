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

def norm_params(name, samples, worker_num=80, **kwargs):
    """Normalization Parameters
    (means, vars)
    """
    os.makedirs(ANALY_PATH, exist_ok=True)
    
    print("Calculating [{}] Means...".format(name))

    manager = Manager(name="Get Means [{}]".format(name),
                      mapper=analysis.sample_sum,
                      reducer=lambda results: [np.sum(results, axis=0)],
                      retries=10,
                      **kwargs
                      )
    manager.hire(worker_num=worker_num)
    print("Assembling Tasks")
    tasks = []
    for _sample in samples:
        tasks.append({"sample": _sample})
    print("Lanuching Jobs")
    sums, nums = manager.launch(tasks=tasks, enable_tqdm=True)[0]
    means = sums / nums

    print("Means", means)
    dump_file = os.path.join(ANALY_PATH, name + ".means")
    with open(dump_file, "wb") as f:
        pickle.dump(means, f)

    print("Calculating [{}] RSSes...".format(name))

    manager = Manager(name="Get RSSes [{}]".format(name),
                      mapper=analysis.sample_rss,
                      reducer=lambda results: [np.sum(results, axis=0)],
                      retries=10,
                      **kwargs
                      )
    manager.hire(worker_num=worker_num)
    print("Assembling Tasks")
    tasks = []
    for _sample in samples:
        tasks.append({"sample": _sample, "means": means})
    print("Lanuching Jobs")
    rsses, nums = manager.launch(tasks=tasks, enable_tqdm=True)[0]
    stds = np.sqrt(rsses / nums)

    print("Stds", stds)
    dump_file = os.path.join(ANALY_PATH, name + ".stds")
    with open(dump_file, "wb") as f:
        pickle.dump(stds, f)



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

    norm_params(name, samples, **kwargs)

if __name__ == "__main__":
    print(sys.argv)
    for _i in range(1, len(sys.argv)):
        main(sys.argv[_i])
