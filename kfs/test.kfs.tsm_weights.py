import os
import time
import json
import shutil
import argparse

import torch
import torchstream

import cfgs
import utils
import kfs

test_log_str = "Testing:[{:4d}/{:4d}]  " + \
               "BatchTime:{batch_time.val:6.2f}({batch_time.avg:6.2f}),  " + \
               "DataTime:{data_time.val:6.2f}({data_time.avg:6.2f}),  " + \
               "Loss:{loss_meter.val:7.3f}({loss_meter.avg:7.3f}),  " + \
               "Prec@1:{top1_meter.val:7.3f}({top1_meter.avg:7.3f}),  " + \
               "Prec@5:{top5_meter.val:7.3f}({top5_meter.avg:7.3f})"

def test(device, loader, model, criterion, 
         log_str=test_log_str, log_interval=20, **kwargs):
    from train import validate
    validate(device, loader, model, criterion, log_str, log_interval, **kwargs)


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

    model = kfs.Wrapper(**configs["model"]["argv"])

    # load TSM model
    checkpoint = torch.load(args.weights)
    model_state_dict = checkpoint["model_state_dict"]
    old_keys = list(model_state_dict.keys())
    for old_key in old_keys:
        new_key = old_key.replace("module.", "")
        val = model_state_dict[old_key]
        del model_state_dict[old_key]
        model_state_dict[new_key] = val
    model.classifier.load_state_dict(model_state_dict)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=configs["gpus"])


    criterion = cfgs.config2criterion(configs["criterion"])
    criterion.to(device)


    test(device, test_loader, model, criterion)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Video Recognition Template")
    # configuration file
    parser.add_argument("config", type=str,
                        help="path to configuration file")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--gpus", nargs='+', type=int, default=None)

    args = parser.parse_args()

    main(args)
