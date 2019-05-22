"""
"""
import numpy as np
import matplotlib.pyplot as plt


from torchstream.datasets import analysis
from torchstream.datasets.utils.mapreduce import Manager
from torchstream.datasets.metadata.collect import collect_samples


def len_hist(name, samples, worker_num=16, **kwargs):
    """
    """
    manager = Manager(name="Get Lenght Hist [{}]".format(name),
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


# def get_norm_params(name, samples, worker_num=16, **kwargs):
#     """Normalization Parameters
#     (means, vars)
#     """
#     pass


def main(name):

    import importlib
    metaset = "torchstream.datasets.metadata.metasets.{}".format(name)
    metaset = importlib.import_module(metaset)

    kwargs = {
        "root" : metaset.JPG_DATA_PATH,
        "layout" : metaset.__layout__,
        "lbls" : metaset.__LABELS__,
        "annots" : metaset.__ANNOTATIONS__,
        # "tmpl" : "{0:05d}",
        # "offset": 1, 
        "mod" : "RGB",
        "ext" : "jpg",
    }
    
    print("Collecting Metadatas")
    samples = collect_samples(**kwargs)

    len_hist(name, samples, **kwargs)

if __name__ == "__main__":
    main("sth_sth_v2")
