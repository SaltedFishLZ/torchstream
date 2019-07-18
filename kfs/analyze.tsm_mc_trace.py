import os
import sys
import pickle

import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

    flag = True
    good_indices_num = []
    sample_num = corrects.size(0)
    for sample_id in range(sample_num):
        good_indices = []
        for i in range(corrects[sample_id].size(0)):
            if corrects[sample_id][i].item():
                good_indices.append(indices[sample_id][i])
                # print(indices[sample_id][i])
        # if flag and (len(good_indices) > 10 and len(good_indices) < 20):
        #     for idx in good_indices:
        #         print(idx)
        #      flag = False
        good_indices_num.append(len(good_indices))
    
    print("Done")
    plt.hist(good_indices_num, bins=25)
    # plt.show()
    plt.savefig("mc-analysis.png", bbox_inches="tight")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        trace_dir = sys.argv[1]
        main(trace_dir)
    else:
        print("cmd <trace_dir>")
