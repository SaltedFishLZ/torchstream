import os
import sys
import pickle
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from pareto import update_pareto_list

def get_result(trace_root, num_samples, thresholds):
    """
    """
    probabilities = []
    corrects = []
    for f in num_samples:
        trace_dir = os.path.join(trace_root, "{}f_trace".format(f))
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

def param2metrics(trace_root, num_samples, params):
    points = []
    for param in params:
        frames, accuracy = get_result(trace_root, num_samples, param)
        points.append(
                    {
                        "flops": frames,
                        "accuracy": accuracy,
                        "param": param
                    }
                )
    return points

def pareto_explore(trace_root, num_samples):

    params = []
    thresh_list = list(np.linspace(0, 1, num=10))
    for a0 in thresh_list:
        for a1 in thresh_list:
            for a2 in thresh_list:
                param = [a0, a1, a2]
                params.append(param)
    points = param2metrics(trace_root, num_samples, params)

    # get pareto frontier
    pareto_points = []
    for point in points:
        update_pareto_list(pareto_points, point)

    return pareto_points


def main(args):

    # sanity check
    assert isinstance(args.trace_root, str), TypeError("trace_root should be a string")
    if not os.path.exists(args.trace_root):
        raise ValueError("trace_root doesn't exist")

    points = []
    params = []
    if args.param_path is None:
        points = pareto_explore(args.trace_root, [4, 8, 16, 32])
        for point in points:
            params.append(point["param"])
    else:
        assert os.path.exists(args.param_path), ValueError("Parameter files not exist")
        with open(args.param_path, "rb") as f:
            params = pickle.load(f)
            points = param2metrics(args.trace_root, [4, 8, 16, 32], params)

    if args.dump_param is not None:
        with open(args.dump_param, "wb") as f:
            pickle.dump(params, f)


    ax = plt.subplot(1, 1, 1)

    flops = []
    accuracy = []
    for point in points:
        flops.append(point["flops"])
        accuracy.append(point["accuracy"])
    ax.scatter(flops, accuracy)

    # ax.hold()
    ax.plot( 4, 0.3742, "or")
    ax.plot( 8, 0.4152, "or")
    ax.plot(16, 0.4324, "or")
    ax.plot(32, 0.4653, "or")

    plt.show()

    # for point in pareto_points:
    #     print(point)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Cascaded Progressive Sampling Tradeoff")
    parser.add_argument("trace_root", type=str)
    parser.add_argument("--param_path", type=str, default=None)
    parser.add_argument("--dump_param", type=str, default=None)
    args = parser.parse_args()

    main(args)
