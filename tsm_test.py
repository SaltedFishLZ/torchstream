
import os
import time
import copy
import json
import shutil

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_


from torchstream.datasets import HMDB51, UCF101, JesterV1, SomethingSomethingV1, SomethingSomethingV2
import torchstream.transforms.transform as streamtransform

from models import TSN, TSMNet
from transforms.transforms import MultiScaleCrop, RandomSegment, CenterSegment
from train import train, validate
from opts import parser
from utils import save_checkpoint




def main(args):

    with open(args.config, "r") as json_config:
        configs = json.load(json_config)
        model_config = configs["model"]
        model_config["input_size"] = tuple(model_config["input_size"])

    best_prec1 = 0
    cudnn.benchmark = True

    device = torch.device("cuda:0")

    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    train_transforms = streamtransform.Compose([
                                RandomSegment(size=8),
                                MultiScaleCrop(224, [1, .875, .75, .66]),
                                streamtransform.RandomHorizontalFlip(),
                                streamtransform.ToTensor(),
                                streamtransform.VideoNormalize(mean=input_mean, std=input_std)
                                ])

    val_transforms = streamtransform.Compose([
                                CenterSegment(size=8),
                                streamtransform.Resize(256),
                                streamtransform.CenterCrop(224),
                                streamtransform.RandomHorizontalFlip(),
                                streamtransform.ToTensor(),
                                streamtransform.VideoNormalize(mean=input_mean, std=input_std)
                                ])

    dataset_config = configs["dataset"]
    if dataset_config["name"] == "hmdb51":
        train_dataset = HMDB51(train=True, transform=train_transforms)
        val_dataset = HMDB51(train=False, transform=val_transforms)
    elif dataset_config["name"] == "ucf101":
        train_dataset = UCF101(train=True, transform=train_transforms)
        val_dataset = UCF101(train=False, transform=val_transforms)
    elif dataset_config["name"] == "jester_v1":
        train_dataset = JesterV1(train=True, transform=train_transforms)
        val_dataset = JesterV1(train=False, transform=val_transforms)
    elif dataset_config["name"] == "sth_sth_v1":
        train_dataset = SomethingSomethingV1(train=True, transform=train_transforms)
        val_dataset = SomethingSomethingV1(train=False, transform=val_transforms)
    elif dataset_config["name"] == "sth_sth_v2":
        train_dataset = SomethingSomethingV2(train=True, transform=train_transforms)
        val_dataset = SomethingSomethingV2(train=False, transform=val_transforms)
    else:
        raise ValueError

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU



    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    model = TSMNet(**model_config)

    ## load checkpoint
    checkpoint = torch.load(configs["checkpoint"])
    model_state_dict = checkpoint["state_dict"]

    old_keys = list(model_state_dict.keys())
    for key in old_keys:
        if "conv1.net" in key:
            val = model_state_dict[key]
            del model_state_dict[key]
            new_key = key.replace('conv1.net', 'conv1.conv')
            model_state_dict[new_key] = val
        if "new_fc" in key:
            val = model_state_dict[key]
            del model_state_dict[key]
            new_key = key.replace('new_fc', 'base_model.fc.fc')
            model_state_dict[new_key] = val
    model.load_state_dict(model_state_dict)
    torch.nn.init.xavier_uniform(model.base_model.fc.fc.weight.data)
    torch.nn.init.xavier_uniform(model.base_model.fc.fc.bias.data)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)



    for epoch in range(args.start_epoch, args.epochs):
        
        ## train for one epoch
        train(device, train_loader, model, criterion, optimizer, epoch)

        ## evaluate on validation set
        prec1 = validate(device, val_loader, model, criterion, epoch)

        # remember best prec@1 
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1) 
        output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
        print(output_best)

        # save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(state={'epoch': epoch,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'best_prec1': best_prec1},
                        is_best=is_best,
                        path="checkpoints",
                        prefix="test")




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
