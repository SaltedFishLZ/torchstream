""""Get spatial resolution distribution histogram
"""
import os
import pickle
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from torchstream.io.analysis import datapoint_hxw
from torchstream.utils.mapreduce import Manager


FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(__file__)
ANALY_PATH = os.path.join(DIR_PATH, ".analyzed.d")

matplotlib.use("pdf")

parser = argparse.ArgumentParser(description="Video Shape Histogram")
parser.add_argument("root", type=str,
                    help="path to dataset root")
parser.add_argument("datapoints", type=str,
                    help="path to dataset root")
parser.add_argument("--name", default=None, type=str,
                    help="histogram file name")
parser.add_argument("--workers", default=16, type=int,
                    help="number of processes")


def get_shape_hist(name, root, datapoints, worker_num, **kwargs):
    os.makedirs(ANALY_PATH, exist_ok=True)

    print("*" * 80)
    manager = Manager(name=name, mapper=datapoint_hxw,
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
    shapes = manager.launch(tasks=tasks, progress=True)

    heights = []
    widths = []
    for hxw in shapes:
        heights.append(hxw[0])
        widths.append(hxw[1])

    nphist = np.histogram2d(heights, widths)
    print("Numpy Hist")
    print("Counts")
    print(nphist[0])
    print("Bins (h & w)")
    print(nphist[1][0])
    print(nphist[1][1])

    plt.hist2d(heights, widths, normed=True)
    plt.savefig(os.path.join(ANALY_PATH,
                             name + ".shape.dist.density.pdf"),
                bbox_inches="tight")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.name is None:
        args.name = args.datapoints.split('/')[-1]

    with open(args.datapoints, "rb") as fin:
        datapoints = pickle.load(fin)

    get_shape_hist(name=args.name,
                   root=args.root,
                   datapoints=datapoints,
                   worker_num=args.workers)
