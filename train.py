"""
"""
import os
import time

import torch

import utils
from validate import validate

train_log_str = "Epoch: [{:3d}][{:4d}/{:4d}], lr: {lr:5.5f}\t" + \
                "BatchTime {batch_time.val:6.2f} ({batch_time.avg:6.2f})\t" + \
                "DataTime {data_time.val:6.2f} ({data_time.avg:6.2f})\t" + \
                "Loss {loss_meter.val:6.3f} ({loss_meter.avg:6.3f})\t" + \
                "Prec@1 {top1_meter.val:6.3f} ({top1_meter.avg:6.3f})\t" + \
                "Prec@5 {top5_meter.val:6.3f} ({top5_meter.avg:6.3f})"


def train_epoch(device, loader, model, criterion, optimizer, epoch,
                log_str=train_log_str, log_interval=20, **kwargs):

    batch_time = utils.Meter()
    data_time = utils.Meter()
    loss_meter = utils.Meter()
    top1_meter = utils.Meter()
    top5_meter = utils.Meter()

    metric = utils.ClassifyAccuracy(topk=(1, 5))

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(loader):        
        input = input.to(device)
        target = target.to(device)

        ## measure extra data loading time
        data_time.update(time.time() - end)

        ## forward
        output = model(input)
        loss = criterion(output, target)

        ## calculate accuracy
        accuracy = metric(output.data, target)
        prec1 = accuracy[1]
        prec5 = accuracy[5]

        ## update statistics
        loss_meter.update(loss, input.size(0))
        top1_meter.update(prec1, input.size(0))
        top5_meter.update(prec5, input.size(0))

        ## backward
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        ## measure elapsed time on GPU
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_interval == 0:
            print(log_str.format(epoch, i, len(loader),
                                 batch_time=batch_time,
                                 data_time=data_time,
                                 loss_meter=loss_meter,
                                 top1_meter=top1_meter,
                                 top5_meter=top5_meter,
                                 lr=optimizer.param_groups[-1]['lr']))



def train(device, train_loader, val_loader,
          model, criterion, optimizer, epochs,
          train_log_str=train_log_str, val_log_str=val_log_str,
          log_interval=20,
          **kwargs):

    best_prec1 = 0
    start_epoch = 0
    if "resume" in kwargs:
        resume_config = kwargs["resume"]
        ckpt_path = os.path.join(resume_config["dir_path"],
                                 resume_config["pth_name"])
        ckpt_path = os.path.realpath(ckpt_path)
        ckpt_path = os.path.expandvars(ckpt_path)
        ckpt_path = os.path.expanduser(ckpt_path)
        checkpoint = torch.load(ckpt_path)
        best_prec1 = checkpoint["best_prec1"]
        start_epoch = checkpoint["epoch"]
        print("Resume from epoch [{}], best prec1 [{}]".format(start_epoch, best_prec1))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])

    backup_config = None
    if "backup" in kwargs:
        backup_config = kwargs["backup"]

    for epoch in range(start_epoch, epochs):
        ## train for one epoch
        train_epoch(device=device, loader=train_loader, model=model,
                    criterion=criterion, optimizer=optimizer,
                    epoch=epoch)

        ## evaluate on validation set
        prec1 = validate(device=device, loader=val_loader, model=model,
                         criterion=criterion, epoch=epoch)

        # remember best prec@1 
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1) 
        print("Best Prec@1: %.3f\n" % (best_prec1))

        # save checkpoint
        if backup_config is not None:
            dir_path = backup_config["dir_path"]
            pth_name = backup_config["pth_name"]
            os.makedirs(dir_path, exist_ok=True)
            utils.save_checkpoint(state={
                                    "epoch": epoch,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "best_prec1": best_prec1
                                    },
                                is_best=is_best,
                                dir_path=dir_path,
                                pth_name=pth_name)
