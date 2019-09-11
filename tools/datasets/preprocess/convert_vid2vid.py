"""Preprocess datasets
"""
import copy
import pickle
import argparse

from torchstream.io.conversion import vid2vid
from torchstream.io.datapoint import DataPoint
from torchstream.utils.mapreduce import Manager

parser = argparse.ArgumentParser(description="Convert Dataset (vid2vid)")
parser.add_argument("datapoints", type=str, help="path to source datapoints")
parser.add_argument("dst", type=str, help="destination dataset root")
parser.add_argument("--ext", type=str, help="destination dataset video ext")
parser.add_argument("--workers", type=int, default=32, help="worker number")


def dataset_vid2vid(name, datapoints, dst_root, dst_ext="avi",
                    workers=16, **kwargs):
    """transform video fils
    """
    manager = Manager(name="converting dataset [{}]".format(name),
                      mapper=vid2vid,
                      retries=10,
                      **kwargs)
    manager.hire(workers=workers)
    tasks = []

    for src in datapoints:
        assert isinstance(src, DataPoint), TypeError
        dst = copy.deepcopy(src)
        dst.root = dst_root
        dst.ext = dst_ext
        dst._path = dst.path
        tasks.append({"src": src, "dste": dst})

    successes = manager.launch(tasks=tasks, enable_tqdm=True)

    # remove failed videos from datapoints
    clean_datapoints = []
    for idx, success in enumerate(successes):
        clean_datapoints.append(datapoints[idx])

    return successes, clean_datapoints


if __name__ == "__main__":
    args = parser.parse_args()
    src_datapoints = []
    with open(args.datapoints, "rb") as fin:
        src_datapoints = pickle.load(fin)
    assert isinstance(src_datapoints, list), TypeError
    assert isinstance(src_datapoints[0], DataPoint), TypeError

    successes, dst_datapoints = dataset_vid2vid(
        name=src_datapoints[0].root,
        datapoints = src_datapoints,
        dst_root=args.dst_root,
        dst_ext=args.dst_ext,
        workers=args.workers
    )

    print("Total Videos: [{}]".format(len(src_datapoints)))
    failures = 0
    for i, dp in enumerate(src_datapoints):
        print("Failure: [{}]".format(i))
        print(dp)
        failures += 1
    print("Total Failures: [{}]".format(failures))

