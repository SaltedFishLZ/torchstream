""""Get # samples per class distribution histogram
"""
import os
import pickle
import argparse
import collections

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from torchstream.io.datapoint import DataPoint


FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(__file__)
ANALY_PATH = os.path.join(DIR_PATH, ".analyzed.d")

matplotlib.use("pdf")

parser = argparse.ArgumentParser(description="# Samples Per Class Histogram")
parser.add_argument("datapoints", type=str,
                    help="path to datapoint file")
parser.add_argument("--name", default=None, type=str,
                    help="histogram file name")


def get_class_hist(name, datapoints):
    os.makedirs(ANALY_PATH, exist_ok=True)

    print("*" * 80)
    print(name)
    print("*" * 80)

    counter = collections.Counter()

    for datapoint in datapoints:
        assert isinstance(datapoint, DataPoint), TypeError
        counter[datapoint.label] += 1

    labels, values = zip(*counter.items())
    values = np.array(list(values))
    indexes = np.arange(len(labels))
    width = 1

    print("Mean", np.mean(values))
    print("Std", np.std(values))
    print("Max", np.max(values))
    print("Min", np.min(values))
    print("Median", np.median(values))

    plt.figure(figsize=(0.5 * len(labels), 6))
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels, rotation="vertical")
    plt.savefig(os.path.join(ANALY_PATH,
                             name + ".class.dist.pdf"),
                bbox_inches="tight")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.name is None:
        args.name = args.datapoints.split('/')[-1]

    with open(args.datapoints, "rb") as fin:
        datapoints = pickle.load(fin)

    get_class_hist(name=args.name,
                   datapoints=datapoints)
