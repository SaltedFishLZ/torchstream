import os
import time
import json
import shutil
import argparse

import torch

import cfgs
import utils

from validate import validate



def main(args):

    ## parse configurations
    configs = {}
    with open(args.config, "r") as json_config:
        configs = json.load(json_config)
    if args.gpus is not None:
        configs["gpus"] = args.gpus
    else:
        configs["gpus"] = list(range(torch.cuda.device_count()))
    device = torch.device("cuda:0")

    test_dataset = cfgs.config2dataset(configs["test_dataset"])


    configs["test_loader"]["dataset"] = test_dataset
    test_loader = cfgs.config2dataloader(configs["test_loader"])

    model = cfgs.config2model(configs["model"])
    model.to(device)
    # Add DataParallel Wrapper
    if args.gpus is not None:
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
    else:
        model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.weights)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)

    configs["optimizer"]["argv"]["params"] = model.parameters()
    optimizer = cfgs.config2optimizer(configs["optimizer"])

    criterion = cfgs.config2criterion(configs["criterion"])
    criterion.to(device)


    validate(device, test_loader,
          model, criterion, optimizer, **(configs["test"]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Video Recognition Template")
    # configuration file
    parser.add_argument("config", type=str,
                        help="path to configuration file")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--gpus", nargs='+', type=int, default=None)
    
    args = parser.parse_args()

    main(args)
