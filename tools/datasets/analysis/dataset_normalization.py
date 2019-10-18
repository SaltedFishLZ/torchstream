"""Get CV normalization parameters:
channel-wise means
channel-wise stds
"""
import os
import pickle
import argparse

import numpy as np

from torchstream.io.analysis import datapoint_sum, datapoint_rss
from torchstream.utils.mapreduce import Manager


FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(__file__)
ANALY_PATH = os.path.join(DIR_PATH, ".analyzed.d")

parser = argparse.ArgumentParser(description="Video Pixel Normalization")
parser.add_argument("root", type=str,
                    help="path to dataset root")
parser.add_argument("datapoints", type=str,
                    help="path to dataset root")
parser.add_argument("--name", default=None, type=str,
                    help="histogram file name")
parser.add_argument("--workers", default=32, type=int,
                    help="number of processes")


def get_norm_params(name, root, datapoints, worker_num, **kwargs):
    os.makedirs(ANALY_PATH, exist_ok=True)

    print("*" * 80)
    print("pixel meam")
    manager = Manager(name=name,
                      mapper=datapoint_sum,
                      reducer=lambda results: [np.sum(results, axis=0)],
                      retries=10, **kwargs)
    manager.hire(worker_num=worker_num)
    print("*" * 80)

    print("assembling tasks")
    tasks = []
    for datapoint in datapoints:
        tasks.append({"datapoint": datapoint})

    print("lanuching jobs")
    results = manager.launch(tasks=tasks, progress=True)

    sums, nums = results[0]
    means = sums / nums

    print("means: ", means)
    dump_file = os.path.join(ANALY_PATH, name + ".means")
    with open(dump_file, "wb") as f:
        pickle.dump(means, f)

    print("*" * 80)
    print("pixel std")
    manager = Manager(name=name,
                      mapper=datapoint_rss,
                      reducer=lambda results: [np.sum(results, axis=0)],
                      retries=10, **kwargs)
    manager.hire(worker_num=worker_num)
    print("*" * 80)

    print("assembling tasks")
    tasks = []
    for datapoint in datapoints:
        tasks.append({"datapoint": datapoint})

    print("lanuching jobs")
    results = manager.launch(tasks=tasks, progress=True)

    rsses, nums = results[0]
    stds = np.sqrt(rsses / nums)

    print("stds: ", stds)
    dump_file = os.path.join(ANALY_PATH, name + ".stds")
    with open(dump_file, "wb") as f:
        pickle.dump(stds, f)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.name is None:
        args.name = args.datapoints.split('/')[-1]

    with open(args.datapoints, "rb") as fin:
        datapoints = pickle.load(fin)

    get_norm_params(name=args.name,
                    root=args.root,
                    datapoints=datapoints,
                    worker_num=args.workers)
