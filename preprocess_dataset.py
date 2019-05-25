"""Preprocess datasets
"""

from torchstream.datasets import preprocess
from torchstream.datasets.utils.mapreduce import Manager
from torchstream.datasets.metadata.collect import collect_samples

def transform_videos(name, samples, dst_root, ext="avi",
                 worker_num=16, **kwargs):
    """transform video format
    """
    manager = Manager(name="Slicing Dataset [{}]".format(name),
                      mapper=preprocess.vid2vid,
                      retries=10,
                      **kwargs
                      )
    manager.hire(worker_num=worker_num)
    tasks = []
    for src_sample in samples:
        dst_sample = src_sample.root_migrated(dst_root)
        dst_sample = dst_sample.extension_migrated(ext=ext)
        tasks.append({"src_sample":src_sample, "dst_sample":dst_sample})
    successes = manager.launch(tasks=tasks, enable_tqdm=True)
    print(successes)

def slice_videos(name, samples, dst_root, ext="jpg",
                 worker_num=16, **kwargs):
    """ Slice videos into images
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
    successes = manager.launch(tasks=tasks, enable_tqdm=True)
    print(successes)


def main(name):
    """
    """
    import importlib
    metaset = "torchstream.datasets.metadata.metasets.{}".format(name)
    metaset = importlib.import_module(metaset)

    kwargs = {
        "root" : metaset.AVI_DATA_PATH,
        "layout" : metaset.__layout__,
        "lbls" : metaset.__LABELS__,
        "mod" : "RGB",
        "ext" : "avi",
    }

    if hasattr(metaset, "__ANNOTATIONS__"):
        kwargs["annots"] = metaset.__ANNOTATIONS__

    if hasattr(metaset, "JPG_FILE_TMPL"):
        kwargs["tmpl"] = metaset.JPG_FILE_TMPL

    if hasattr(metaset, "JPG_IDX_OFFSET"):
        kwargs["offset"] = metaset.JPG_IDX_OFFSET

    samples = collect_samples(**kwargs)

    # transform_videos(name, samples, dst_root=metaset.AVI_DATA_PATH)
    slice_videos(name, samples, dst_root=metaset.JPG_DATA_PATH)

if __name__ == "__main__":
    import  sys
    print(sys.argv)
    for _i in range(1, len(sys.argv)):
        main(sys.argv[_i])
