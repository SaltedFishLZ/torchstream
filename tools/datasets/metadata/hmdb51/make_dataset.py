"""Collecting DataPoints for different configurations of HMDB51
and dump the pickle file
"""
import os
import argparse

from torchstream.utils.metadata import collect_folder

import label
import split

parser = argparse.ArgumentParser()
parser.add_argument("root", type=str, help="path to dataset root")
parser.add_argument("--ext", default="avi", type=str,
                    help="dataset file extension")
parser.add_argument("--split", default=1, type=int,
                    help="dataset splitting plan")


def main(args):
    # collecting data points
    all_datapoints = collect_folder(
        root=args.root, ext=args.ext,
        annotations=label.class_to_idx
    )
    print(len(all_datapoints))

    train_set_names = split.train_sample_names(split_num=args.split)
    test_set_names = split.test_sample_names(split_num=args.split)

    # filter data points
    training_datapoints = []
    testing_datapoints = []
    for dp in all_datapoints:
        if dp.name in train_set_names:
            training_datapoints.append(dp)
        elif dp.name in test_set_names:
            testing_datapoints.append(dp)
    print(len(train_set_names))
    print(len(test_set_names))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)