"""Collecting DataPoints for different configurations of
the `Something Something v1` dataset  and dump the pickle file
"""
import os
import pickle
import argparse

from torchstream.io.datapoint import UNKNOWN_LABEL
from torchstream.utils.metadata import collect_flat

import annotation

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

parser = argparse.ArgumentParser()
parser.add_argument("root", type=str, help="path to dataset root")
parser.add_argument("--ext", default="jpg", type=str,
                    help="dataset file extension")
parser.add_argument("--fpath-offset", default=1, type=int,
                    help="frame index offset")
parser.add_argument("--fpath-tmpl", default="{:05d}", type=str,
                    help="frame path template")


def main(args):

    # collecting all data points
    all_datapoints = collect_flat(
        root=args.root, ext=args.ext,
        annotations=annotation.full_annot_dict,
        fpath_offset=args.fpath_offset,
        fpath_tmpl=args.fpath_tmpl
    )

    # filter data points
    training_datapoints = []
    validation_datapoints = []
    testing_datapoints = []
    for dp in all_datapoints:
        if dp.name in annotation.train_annot_dict:
            training_datapoints.append(dp)
        elif dp.name in annotation.val_annot_dict:
            validation_datapoints.append(dp)
        elif dp.label == UNKNOWN_LABEL:
            testing_datapoints.append(dp)
        else:
            raise ValueError

    # dump files
    training_pickle = os.path.join(
        DIR_PATH,
        "sthsthv1_{}_training.pkl".format(args.ext)
    )
    with open(training_pickle, "wb") as fout:
        pickle.dump(training_datapoints, fout)

    validation_pickle = os.path.join(
        DIR_PATH,
        "sthsthv1_{}_validation.pkl".format(args.ext)
    )
    with open(validation_pickle, "wb") as fout:
        pickle.dump(validation_datapoints, fout)

    testing_pickle = os.path.join(
        DIR_PATH,
        "sthsthv1_{}_testing.pkl".format(args.ext)
    )
    with open(testing_pickle, "wb") as fout:
        pickle.dump(testing_datapoints, fout)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
