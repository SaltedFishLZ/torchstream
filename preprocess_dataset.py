"""Preprocess datasets
"""

from torchstream.datasets import preprocess
from torchstream.datasets.utils.mapreduce import Manager
from torchstream.datasets.metadata.collect import collect_samples

def slice_videos(name, samples, dst_root, ext="jpg",
                 worker_num=16, **kwargs):
    """
    """
    manager = Manager(name="Slicing Dataset [{}]".format(name),
                      mapper=preprocess.vid2seq,
                      retries=10,
                      **kwargs
                      )
    manager.hire(worker_num=worker_num)
    tasks = []
    for src_sample in samples:
        dst_sample = src_sample.root_migrated(dst_root)
        dst_sample = dst_sample.extension_migrated(ext=ext)
        tasks.append({"src_sample":src_sample, "dst_sample":dst_sample})
    lens = manager.launch(tasks=tasks, enable_tqdm=True)
    print(lens)
    # result = np.histogram(lens, bins=1)
    # print(result)


def main(name):
    """
    """
    import importlib
    name = "weizmann"
    metaset = "torchstream.datasets.metadata.metasets.{}".format(name)
    metaset = importlib.import_module(metaset)

    kwargs = {
        "root" : metaset.AVI_DATA_PATH,
        "layout" : metaset.__layout__,
        "lbls" : metaset.__LABELS__,
        "mod" : "RGB",
        "ext" : "avi",
    }
    samples = collect_samples(**kwargs)

    slice_videos(name, samples, dst_root=metaset.JPG_DATA_PATH)

if __name__ == "__main__":
    main("sth_sth_v2")
