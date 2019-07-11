"""
"""
import os
import time
import json
import random
import shutil
import argparse

import torch
import torchstream
import numpy as np

import cfgs
import utils
from models import FrameQualityDiscriminator

def gen_indices_random(ni, no, chances, output_type="tensor_onehot"):
    """Generate selected indices randomly
    """
    SUPPORTED_OUTPUTS = [
        "list", "ndarray", "tensor", "tensor_onehot"
        ]
    assert output_type in SUPPORTED_OUTPUTS, NotImplementedError

    candidates = list(range(ni))
    # padding
    while len(candidates) < no:
        candidates.append(candidates[-1])

    indices = []
    for i in range(chances):
        # sample without replacement
        idx = random.sample(candidates, no)
        idx.sort()
        indices.append(idx)

    if output_type == "list":
        return indices
    elif output_type == "ndarray":
        return np.array(indices)
    elif output_type == "tensor":
        return torch.Tensor(indices).long()
    elif output_type == "tensor_onehot":
        indices = torch.Tensor(indices).long()
        indices_onehot = torch.zeros(chances, ni)
        indices_onehot.scatter_(1, indices.long(), 1)
        return indices_onehot


def cherrypick_frames(device, input, discriminator):
    """
    Args:
        input [C][T][H][W]
    """
    discriminator.eval()

    chances = 50
    index = gen_indices_random(ni=16, no=8, chances=chances)
    output = None
    with torch.no_grad():
        C, T, H, W = input.size()
        input = input.to(device)
        input = input.unsqueeze(dim=0)
        input = input.expand(chances, C, T, H, W)
        output = discriminator((input, index))
        output = torch.nn.functional.softmax(output, dim=-1)

    index_quality = output[:, 1]
    index_selection = torch.argmax(index_quality)
    print(index_selection)
    print(index_quality)


def main(args):

    # parse configurations
    configs = {}
    with open(args.config, "r") as json_config:
        configs = json.load(json_config)
    if args.gpus is not None:
        configs["gpus"] = args.gpus
    else:
        configs["gpus"] = list(range(torch.cuda.device_count())) 

    device = torch.device("cuda:0")


    # -------------------------------------------------------- #
    #          Construct Datasets & Dataloaders                #
    # -------------------------------------------------------- #

    test_transforms = []
    for _t in configs["test_transforms"]:
        test_transforms.append(cfgs.config2transform(_t))
    test_transforms = torchstream.transforms.Compose(
        transforms=test_transforms
        )

    configs["test_dataset"]["argv"]["transform"] = test_transforms
    test_dataset = cfgs.config2dataset(configs["test_dataset"])

    configs["test_loader"]["dataset"] = test_dataset
    test_loader = cfgs.config2dataloader(configs["test_loader"])


    # -------------------------------------------------------- #
    #          Construct Network & Load Weights                #
    # -------------------------------------------------------- #

    discriminator = cfgs.config2model(configs["discriminator"])
    discriminator.to(device)

    discriminator = torch.nn.DataParallel(discriminator, device_ids=configs["gpus"])

    # load model
    checkpoint = torch.load(args.weights)
    model_state_dict = checkpoint["model_state_dict"]
    discriminator.load_state_dict(model_state_dict)

    # criterion = cfgs.config2criterion(configs["criterion"])
    # criterion.to(device)

    cherrypick_frames(device, test_dataset[1][0], discriminator)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Video Recognition Template")
    # configuration file
    parser.add_argument("config", type=str,
                        help="path to configuration file")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--gpus", nargs='+', type=int, default=None)

    args = parser.parse_args()

    main(args)
