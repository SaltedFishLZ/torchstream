"""
"""
import os
import sys
import copy
import pickle
import logging
import hashlib
import multiprocessing as mp

import tqdm
import numpy as np
import torch
import torch.utils.data as data

from torchstream.datasets import SomethingSomethingV1

def sample_difficulty():
    pass

# ------------------------------------------------------------------------- #
#                   Main Classes (To Be Used outside)                       #
# ------------------------------------------------------------------------- #

## frame quality for video recognition
class FrameQualityDataset(data.Dataset):
    """Frame Quality Dataset
    """
    def __init__(self, trace_root, chances=50, num_snippets=8, num_frames=16,
                 train=True, transform=None, target_transform=None,
                 **kwargs):
        """
        Args:

        """
        check_str = "s{}.f{}.{}".format(num_snippets, num_frames,
                                        "train" if train else "test")
        assert check_str in trace_root, ValueError("Invalid trace toor")

        self.trace_root = trace_root
        self.chances = chances

        self.video_dataset = SomethingSomethingV1(
            train=train,
            transform=transform,
            target_transform=target_transform,
            **kwargs
            )

        self.indices = None
        self.corrects = None
        for chance in range(chances):

            chance_dir_path = os.path.join(trace_root, "chance{}".format(chance))
            index_file_path = os.path.join(chance_dir_path, "index.pkl")
            correct_file_path = os.path.join(chance_dir_path, "correct.pkl")

            with open(index_file_path, "rb") as f:
                index = pickle.load(f)
                if self.indices is None:
                    self.indices = index.unsqueeze(dim=1)
                else:
                    self.indices = torch.cat((self.indices, index), dim=1)

            with open(correct_file_path, "rb") as f:
                correct = pickle.load(f)
                if self.corrects is None:
                    self.corrects = correct.unsqueeze(dim=1)
                else:
                    self.corrects = torch.cat((self.corrects, correct), dim=1)

    def __len__(self):
        return len(len(self.video_dataset) * self.chances)

    def __getitem__(self, idx):
        """
            [video][chance]
            (index, blob, cid)
        """
        blob, cid = self.video_dataset[idx]

        # return (a [T][H][W][C] ndarray, class id)
        # ndarray may need to be converted to [T][C][H][W] format in PyTorch
        return (blob, cid)


