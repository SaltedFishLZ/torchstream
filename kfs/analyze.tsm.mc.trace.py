import os
import sys
import pickle

import torch


def main(trace_dir):

    indices = None
    corrects = None

    for chance_dir in os.listdir(trace_dir):
        if "chance" not in chance_dir:
            continue

        index_fpath = os.path.join(trace_dir, chance_dir, "index.pkl")
        correct_fpath = os.path.join(trace_dir, chance_dir, "correct.pkl")

        index_pkl = open(index_fpath, "rb")
        index = pickle.load(index_pkl)
        index_pkl.close()

        correct_pkl = open(correct_fpath, "rb")
        correct = pickle.load(correct_pkl)
        correct_pkl.close()

        index = index.unsqueeze(dim=1)
        correct = correct.unsqueeze(dim=1)

        if indices is None:
            indices = index
            corrects = correct
        else:
            indices = torch.cat((indices, index), dim=1)
            corrects = torch.cat((corrects, correct), dim=1)

    print(indices)
    print(corrects)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        trace_dir = sys.argv[1]
        main(trace_dir)
    else:
        print("cmd <trace_dir>")
