"""
"""
import sys

import numpy as np
import matplotlib.pyplot as plt


from torchstream.datasets import analysis
from torchstream.datasets.utils.mapreduce import Manager
from torchstream.datasets.metadata.collect import collect_samples


def len_hist(name, samples, worker_num=40, **kwargs):
    """
    """
    manager = Manager(name="Get Length Hist [{}]".format(name),
                      mapper=analysis.sample_len,
                      retries=10,
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
    
    nphist = np.histogram(lens, bins=10)
    print("Numpy Hist")
    print("Density", nphist[0])
    print("Bins", nphist[1])

    plt.hist(lens, density=True, bins=20)
    plt.show()


def norm_params(name, samples, worker_num=40, **kwargs):
    """Normalization Parameters
    (means, vars)
    """

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
    vars = np.sqrt(sums / nums)

    print("Vars", vars)




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
    
    if hasattr(metaset, "__ANNOTATIONS__"):
        kwargs["annots"] = metaset.__ANNOTATIONS__

    if hasattr(metaset, "JPG_FILE_TMPL"):
        kwargs["tmpl"] = metaset.JPG_FILE_TMPL
    
    if hasattr(metaset, "JPG_IDX_OFFSET"):
        kwargs["offset"] = metaset.JPG_IDX_OFFSET
    
    print("Collecting Metadatas")
    samples = collect_samples(**kwargs)

    # len_hist(name, samples, **kwargs)
    norm_params(name, samples, **kwargs)

if __name__ == "__main__":
    print(sys.argv)
    for _i in range(1, len(sys.argv)):
        main(sys.argv[_i])
