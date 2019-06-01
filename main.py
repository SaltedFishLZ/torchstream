# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import json
import shutil

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_


from torchstream.datasets.hmdb51 import HMDB51
import torchstream.transforms.transform as streamtransform

from models import TSN
from transforms.transforms import MultiScaleCrop
from train import train, validate
from opts import parser
from utils import save_checkpoint




def main(args):

    with open(args.json_config, "r") as json_config:
        configs = json.load(json_config)
        model_config = configs["model"]

    best_prec1 = 0
    cudnn.benchmark = True

    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    train_dataset = HMDB51(train=True, transform=streamtransform.Compose([
                                MultiScaleCrop(224, [1, .875, .75, .66]),
                                streamtransform.RandomHorizontalFlip(),
                                streamtransform.ToTensor(),
                                streamtransform.VideoNormalize(mean=input_mean, std=input_std)
                                ])
                           )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_dataset = HMDB51(train=False, transform=streamtransform.Compose([
                                streamtransform.Resize(256),
                                streamtransform.CenterCrop(224),
                                streamtransform.RandomHorizontalFlip(),
                                streamtransform.ToTensor(),
                                streamtransform.VideoNormalize(mean=input_mean, std=input_std)
                                ])
                           )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    model = TSN(**model_config)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.Adam(model.parameters())


    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()



    for epoch in range(args.start_epoch, args.epochs):
        
        ## train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        ## evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1) 
        output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
        print(output_best)

        # save checkpoint
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