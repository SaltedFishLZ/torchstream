"""Slice videos in certain dataset
"""
import os
import copy
import pickle
import argparse

from torchstream.io.conversion import vid2seq
from torchstream.io.datapoint import DataPoint
from torchstream.io.__support__ import SUPPORTED_IMAGES
from torchstream.utils.mapreduce import Manager

parser = argparse.ArgumentParser(description="Convert Dataset (vid2seq)")
parser.add_argument("src_dp_path", type=str,
                    help="path to source datapoints")
parser.add_argument("--dst_dp_path", type=str, default=None,
                    help="path to destination datapoints")
parser.add_argument("src_root", type=str, help="source dataset root")
parser.add_argument("dst_root", type=str, help="destination dataset root")
parser.add_argument("--dst_ext", type=str, default="jpg",
                    help="destination dataset image sequence file extension")
parser.add_argument("--workers", type=int, default=32, help="worker number")


def dataset_vid2seq(name, src_datapoints, dst_root, dst_ext="jpg",
                    workers=16, **kwargs):
    """slicing video files into frames
    """
    manager = Manager(name="converting dataset [{}]".format(name),
                      mapper=vid2seq,
                      retries=10,
                      **kwargs)
    manager.hire(worker_num=workers)

    tasks = []
    dst_datapoints = []
    for src in src_datapoints:
        assert isinstance(src, DataPoint), TypeError
        dst = copy.deepcopy(src)
        dst.root = dst_root
        dst.ext = dst_ext
        dst._path = dst.path
        os.makedirs(dst._path)
        dst._seq = dst.seq
        dst_datapoints.append(dst)
        tasks.append({"src": src, "dst": dst})

    successes = manager.launch(tasks=tasks, progress=True)

    # remove failed videos from datapoints
    clean_datapoints = []
    for idx, success in enumerate(successes):
        clean_datapoints.append(dst_datapoints[idx])
    dst_datapoints = clean_datapoints

    # count frame numbers
    print("counting frame numbers for each image sequence")
    for datapoint in dst_datapoints:
        datapoint._fcount = datapoint.fcount

    return successes, dst_datapoints


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.dst_ext in SUPPORTED_IMAGES["RGB"], \
        NotImplementedError("Unknown dst_ext [{}]".format(args.dst_ext))
    # test permission first to save time
    test_write_content = [0, 1, 2]
    if args.dst_dp_path is not None:
        with open(args.dst_dp_path, "wb") as fout:
            pickle.dump(test_write_content, fout)

    src_datapoints = []
    with open(args.src_dp_path, "rb") as fin:
        src_datapoints = pickle.load(fin)
    assert isinstance(src_datapoints, list), TypeError
    assert isinstance(src_datapoints[0], DataPoint), TypeError

    successes, dst_datapoints = dataset_vid2seq(
        name=src_datapoints[0].root,
        src_datapoints=src_datapoints,
        dst_root=args.dst_root,
        dst_ext=args.dst_ext,
        workers=args.workers
    )

    failures = 0
    for idx, dp in enumerate(dst_datapoints):
        if not successes[idx]:
            print("Failure: [{}]".format(idx))
            print(dp)
            failures += 1
    print("Total Videos: [{}]".format(len(src_datapoints)))
    print("Total Failures: [{}]".format(failures))

    if args.dst_dp_path is not None:
        with open(args.dst_dp_path, "wb") as fout:
            pickle.dump(dst_datapoints, fout)
