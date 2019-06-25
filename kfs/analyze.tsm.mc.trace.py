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

        index_fpath = os.path.join(trace_dir, chance_dir, "index.pkl", "rb")
        correct_fpath = os.path.join(trace_dir, chance_dir, "correct.pkl", "rb")

        index = pickle.load(index_fpath)
        correct = pickle.load(correct_fpath)

        index = index.unqueeze(dim=1)
        correct = correct.unqueeze(dim=1)

        if indices is None:
            indices = index
            corrects = correct
        else:
            indices = torch.cat((indices, index), dim=1)
            corrects = torch.cat((corrects, correct), dim=1)

    print(indices.size())
    print(corrects.size())


if __name__ == "__main__":
    if len(sys.argv) == 2:
        trace_dir = sys.argv[1]
        main(trace_dir)
    else:
        print("cmd <trace_dir>")