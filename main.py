"""
"""
import os
import time
import json
import shutil
import argparse

import torch

import cfgs
import utils

from train import train
from validate import validate, val_log_str



def main(args):

    ## parse configurations
    configs = {}
    with open(args.config, "r") as json_config:
        configs = json.load(json_config)
    if args.gpus is not None:
        configs["gpus"] = args.gpus
    else:
        configs["gpus"] = list(range(torch.cuda.device_count()))
    configs["train"]["start_epoch"] = 0
    device = torch.device("cuda:0")


    train_dataset = cfgs.config2dataset(configs["train_dataset"])
    val_dataset = cfgs.config2dataset(configs["val_dataset"])

    configs["train_loader"]["dataset"] = train_dataset
    train_loader = cfgs.config2dataloader(configs["train_loader"])

    configs["val_loader"]["dataset"] = val_dataset
    val_loader = cfgs.config2dataloader(configs["val_loader"])

    configs["model"]["argv"]["input_size"] = tuple(configs["model"]["argv"]["input_size"])
    model = cfgs.config2model(configs["model"])
    model.to(device)
    # Add DataParallel Wrapper
    if args.gpus is not None:
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
    else:
        model = torch.nn.DataParallel(model)

    configs["optimizer"]["argv"]["params"] = model.parameters()
    optimizer = cfgs.config2optimizer(configs["optimizer"])

    criterion = cfgs.config2criterion(configs["criterion"])
    criterion.to(device)


    train(device, train_loader, val_loader,
          model, criterion, optimizer, **(configs["train"]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Video Recognition Template")
    # configuration file
    parser.add_argument("config", type=str,
                        help="path to configuration file")
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    
    args = parser.parse_args()

    main(args)
