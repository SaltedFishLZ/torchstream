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

def random_select_list(n, k, sort=True):
    assert n >= k, ValueError
    candidates = list(range(n))
    results = random.sample(candidates, k)
    if sort:
        results.sort()
    return results

def gen_indices_random(fi, fo, n, chances, bind=False, output_type="tensor"):
    """Generate selected indices randomly
    """
    SUPPORTED_OUTPUTS = [
        "list", "ndarray", "tensor", "tensor"
        ]
    assert output_type in SUPPORTED_OUTPUTS, NotImplementedError

    index = []
    if bind:
        index_per_sample = []
        for chance in range(chances):
            index_per_sample.append(random_select_list(n=fi, k=fo))
        index = [index_per_sample, ] * n
    else:
        for i in range(n):
            index_per_sample = []
            for chance in range(chances):
                index_per_sample.append(random_select_list(n=fi, k=fo))
            index.append(index_per_sample)

    if output_type == "list":
        return index
    elif output_type == "ndarray":
        return np.array(index)
    elif output_type == "tensor":
        return torch.Tensor(index).long()

def restricted_random_select_list(n, k):
    assert n >= k, ValueError
    interval = float(n) / float(k)
    offsets = interval * np.array(range(k))

    cursors = []
    while len(cursors) < k:
        cursors.append(random.uniform(0, interval))
    cursors = np.array(cursors)
    
    indices = offsets + cursors
    indices = np.uint(indices)
    indices.sort()
    indices = np.minimum(indices, n - 1)

    return indices

def gen_indices_restricted_random(fi, fo, n, chances, bind=False,
                                  output_type="tensor"):
    SUPPORTED_OUTPUTS = [
        "list", "ndarray", "tensor", "tensor"
        ]
    assert output_type in SUPPORTED_OUTPUTS, NotImplementedError

    index = []
    if bind:
        index_per_sample = []
        for chance in range(chances):
            index_per_sample.append(restricted_random_select_list(n=fi, k=fo))
        index = [index_per_sample, ] * n
    else:
        for i in range(n):
            index_per_sample = []
            for chance in range(chances):
                index_per_sample.append(restricted_random_select_list(n=fi, k=fo))
            index.append(index_per_sample)

    if output_type == "list":
        return index
    elif output_type == "ndarray":
        return np.array(index)
    elif output_type == "tensor":
        return torch.Tensor(index).long()

def cherrypick_frames(device, input, discriminator):
    """
    Args:
        input [N][C][T][H][W]
    """

    N, C, T, H, W = input.size()
    chances = 64


    output = None

    input = input.to(device)
    input = input.unsqueeze(dim=1)
    input = input.expand(N, chances, C, T, H, W)
    input = input.contiguous().view(N * chances, C, T, H, W)

    # gen indices
    index = gen_indices_restricted_random(fi=16, fo=8, n=N, chances=chances)
    index = index.view(N * chances, -1)
    index_onehot = torch.zeros(N * chances, 16)
    index_onehot.scatter_(1, index.long(), 1)
    index_onehot = index_onehot.to(device)

    output = discriminator((input, index_onehot))
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output.view(N, chances, 2)

    index_quality = output[:, :, 1]
    # print("index_quality", index_quality.size())
    # print(index_quality)

    index_selection = torch.argmax(index_quality,dim=1)
    # print("index_selection", index_selection)

    cherrypicked_index = index[index_selection.view(N, 1)]
    cherrypicked_index = cherrypicked_index.squeeze(dim=1)

    return cherrypicked_index

test_log_str = "Testing:[{:4d}/{:4d}]  " + \
               "BatchTime:{batch_time.val:6.2f}({batch_time.avg:6.2f}),  " + \
               "DataTime:{data_time.val:6.2f}({data_time.avg:6.2f}),  " + \
               "Loss:{loss_meter.val:7.3f}({loss_meter.avg:7.3f}),  " + \
               "Prec@1:{top1_meter.val:7.3f}({top1_meter.avg:7.3f}),  " + \
               "Prec@5:{top5_meter.val:7.3f}({top5_meter.avg:7.3f})"

def test(device, loader, discriminator, classifier, criterion,
         log_str=test_log_str, log_interval=20, **kwargs):

    batch_time = utils.Meter()
    data_time = utils.Meter()
    loss_meter = utils.Meter()
    top1_meter = utils.Meter()
    top5_meter = utils.Meter()

    metric = utils.ClassifyAccuracy(topk=(1, 5))

    discriminator.eval()
    classifier.eval()

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):

            N, C, T, H, W = input.size()

            input = input.to(device)
            target = target.to(device)

            # measure extra data loading time
            data_time.update(time.time() - end)

            index = cherrypick_frames(device, input, discriminator)
            input = input.permute(0, 2, 1, 3, 4).contiguous()

            cherrypicked_input = torch.zeros(N, 8, C, H, W)
            for j in range(N):
                cherrypicked_input[j] = input[j][index[j]]

            cherrypicked_input = cherrypicked_input.permute(0, 2, 1, 3, 4).contiguous()

            output = classifier(cherrypicked_input)
            loss = criterion(output, target)

            accuracy = metric(output.data, target)
            prec1 = accuracy[1]
            prec5 = accuracy[5]

            loss_meter.update(loss, input.size(0))
            top1_meter.update(prec1, input.size(0))
            top5_meter.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_interval == 0:
                print(log_str.format(i, len(loader),
                                     batch_time=batch_time,
                                     data_time=data_time,
                                     loss_meter=loss_meter,
                                     top1_meter=top1_meter,
                                     top5_meter=top5_meter))

    print("Results:\n"
          "Prec@1 {top1_meter.avg:5.3f} "
          "Prec@5 {top5_meter.avg:5.3f} "
          "Loss {loss_meter.avg:5.3f}"
          .format(top1_meter=top1_meter,
                  top5_meter=top5_meter,
                  loss_meter=loss_meter))

    return top1_meter.avg

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

    # construct discriminator
    discriminator = cfgs.config2model(configs["discriminator"])
    discriminator.to(device)

    discriminator = torch.nn.DataParallel(discriminator, device_ids=configs["gpus"])

    # load discriminator model
    checkpoint = torch.load(args.discriminator_weights)
    model_state_dict = checkpoint["model_state_dict"]
    discriminator.load_state_dict(model_state_dict)

    # construct classifier
    classifier = cfgs.config2model(configs["classifier"])
    classifier.to(device)

    classifier = torch.nn.DataParallel(classifier, device_ids=configs["gpus"])

    # load discriminator model
    checkpoint = torch.load(args.classifier_weights)
    model_state_dict = checkpoint["model_state_dict"]
    classifier.load_state_dict(model_state_dict)  

    criterion = cfgs.config2criterion(configs["criterion"])
    criterion.to(device)

    test(device=device, loader=test_loader,
         discriminator=discriminator,
         classifier=classifier,
         criterion=criterion)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Video Recognition Template")
    # configuration file
    parser.add_argument("config", type=str,
                        help="path to configuration file")
    parser.add_argument("--discriminator_weights", type=str, default=None)
    parser.add_argument("--classifier_weights", type=str, default=None)
    parser.add_argument("--gpus", nargs='+', type=int, default=None)

    args = parser.parse_args()
    assert args.discriminator_weights is not None, ValueError("Must specify weights")
    assert args.classifier_weights is not None, ValueError("Must specify weights")

    main(args)
