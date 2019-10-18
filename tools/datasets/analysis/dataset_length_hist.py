""""Get temporal duration distribution histogram
"""
import os
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from torchstream.io.analysis import datapoint_len
from torchstream.utils.mapreduce import Manager


FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(__file__)
ANALY_PATH = os.path.join(DIR_PATH, ".analyzed.d")

parser = argparse.ArgumentParser(description="Video Length Histogram")
parser.add_argument("root", type=str,
                    help="path to dataset root")
parser.add_argument("datapoints", type=str,
                    help="path to dataset root")
parser.add_argument("--name", default=None, type=str,
                    help="histogram file name")
parser.add_argument("--workers", default=16, type=int,
                    help="number of processes")


def get_length_hist(name, root, datapoints, worker_num, **kwargs):
    os.makedirs(ANALY_PATH, exist_ok=True)

    print("*" * 80)
    manager = Manager(name=name, mapper=datapoint_len,
                      retries=10, **kwargs)
    manager.hire(worker_num=worker_num)
    print("*" * 80)

    print("assembling tasks...")
    tasks = []
    for datapoint in datapoints:
        # change root
        datapoint.root = root
        datapoint._path = datapoint.path
        tasks.append({"datapoint": datapoint})

    print("lanuching jobs")
    lengths = manager.launch(tasks=tasks, progress=True)

    nphist = np.histogram(lengths)
    print("Numpy Hist")
    print("Counts")
    print(nphist[0])
    print("Bins")
    print(nphist[1])

    plt.hist(lengths)
    plt.savefig(os.path.join(ANALY_PATH,
                             name + ".length.dist.pdf"),
                bbox_inches="tight")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.name is None:
        args.name = args.datapoints.split('/')[-1]

    with open(args.datapoints, "rb") as fin:
        datapoints = pickle.load(fin)

    get_length_hist(name=args.name,
                    root=args.root,
                    datapoints=datapoints,
                    worker_num=args.workers)
