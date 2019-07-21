import os
import pickle

import torch

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
            print("prob", prob)
            probabilities.append(prob)
        with open(correct_path, "rb") as f:
            correct = pickle.load(f)
            correct = correct[0]
            print("correct", correct)
            print(correct.sum())
            corrects.append(correct)
    

def main():
    get_result([4, 8, 16, 32], [0.5, 0.5, 0.5, 0.5])


if __name__ == "__main__":
    main()