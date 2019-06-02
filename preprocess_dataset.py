"""Preprocess datasets
"""

from torchstream.datasets import preprocess
from torchstream.datasets.utils.mapreduce import Manager
from torchstream.datasets.metadata.collect import collect_datapoints

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


def aggregate_frames(name, samples, dst_root, ext="avi",
                     tmpl="{}", offset=0,
                     worker_num=32):
    """frames -> videos
    """
    print("DST PATH", dst_root)
    manager = Manager(name="Aggregating Dataset [{}]".format(name),
                      mapper=preprocess.seq2vid)
    manager.hire(worker_num=worker_num)
    tasks = []
    for src_sample in samples:
        dst_sample = src_sample.root_migrated(dst_root)
        dst_sample = dst_sample.extension_migrated(ext=ext)
        task_dict = {
                "src_sample":src_sample, 
                "dst_sample":dst_sample,
                "tmpl": tmpl, "offset": offset
                }
        tasks.append(task_dict)
    successes = manager.launch(tasks=tasks, enable_tqdm=True)
    print(successes)


def main(name):
    """
    """
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

    samples = collect_datapoints(**kwargs)


    kwargs = {}
    if hasattr(metaset, "JPG_FILE_TMPL"):
        kwargs["tmpl"] = metaset.JPG_FILE_TMPL
    if hasattr(metaset, "JPG_IDX_OFFSET"):
        kwargs["offset"] = metaset.JPG_IDX_OFFSET

    # transform_videos(name, samples, dst_root=metaset.AVI_DATA_PATH)
    # slice_videos(name, samples, dst_root=metaset.JPG_DATA_PATH)
    aggregate_frames(name, samples, dst_root=metaset.AVI_DATA_PATH, **kwargs)

if __name__ == "__main__":
    import  sys
    print(sys.argv)
    for _i in range(1, len(sys.argv)):
        main(sys.argv[_i])
