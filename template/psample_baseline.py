import os
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

def get_result(num_samples, thresholds):
    """
    """
    probabilities = []
    corrects = []
    for f in num_samples:
        trace_dir = "{}f_trace".format(f)
        prob_path = os.path.join(trace_dir, "trace_prob.pkl")
        correct_path = os.path.join(trace_dir, "trace_correct.pkl")
        with open(prob_path, "rb") as f:
            prob = pickle.load(f)
            prob = torch.max(prob, dim=-1)[0]
            # print("prob", prob)
            probabilities.append(prob)
        with open(correct_path, "rb") as f:
            correct = pickle.load(f)
            correct = correct[0]
            # print("correct")
            # print(float(correct.sum())/correct.size(0))
            corrects.append(correct)

    N = correct.size(0)
    masks = []
    running_mask = torch.ones(N, dtype=torch.uint8)
    for i in range(len(thresholds)):
        mask = running_mask * (probabilities[i] > thresholds[i])
        masks.append(mask)
        running_mask = running_mask - mask
    masks.append(running_mask)

    final_correct = torch.zeros(N, dtype=torch.uint8)
    for i in range(len(masks)):
        final_correct += masks[i] * corrects[i]
    accuracy = (float(final_correct.sum()) / N)

    final_frames = torch.zeros(N, dtype=torch.uint8)
    for i in range(len(masks)):
        final_frames += masks[i] * num_samples[i]
    frames = float(torch.mean(final_frames.float()))

    return(frames, accuracy)

def main():
    points = []
    thresh_list = list(np.linspace(0, 1, num=10))
    print(thresh_list)
    for a0 in thresh_list:
        for a1 in thresh_list:
            for a2 in thresh_list:
                points.append(get_result([4, 8, 16, 32], [a0, a1, a2]))
    frames, accuracy = zip(*points)

    ax = plt.subplot(1, 1, 1)
    ax.scatter(frames, accuracy)
    # ax.hold()
    ax.plot(8, 0.4007, "or")
    ax.plot(16, 0.4324, "or")
    ax.plot(32, 0.4653, "or")

    plt.show()

if __name__ == "__main__":
    main()
