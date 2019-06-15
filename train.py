"""
"""
import os
import time

import torch

import utils
from validate import validate, val_log_str

train_log_str = "Epoch: [{:3d}][{:4d}/{:4d}], lr: {lr:.5f}\t" + \
                "BatchTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t" + \
                "DataTime {data_time.val:.3f} ({data_time.avg:.3f})\t" + \
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t" + \
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t" + \
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})"

def train_epoch(device, loader, model, criterion, optimizer, epoch,
                log_str, log_interval=20, **kwargs):

    batch_time = utils.Meter()
    data_time = utils.Meter()
    losses = utils.Meter()
    top1 = utils.Meter()
    top5 = utils.Meter()

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
        losses.update(loss, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        ## backward
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        ## measure elapsed time on GPU
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_interval == 0:
            print(log_str.format(epoch, i, len(loader),
                                 batch_time=batch_time, data_time=data_time,
                                 loss=losses, top1=top1, top5=top5, 
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
        optimizer.load_stat_dict(checkpoint["optimizer_state_dict"])
        optimizer.load_stat_dict(checkpoint["optimizer_state_dict"])

    backup_config = None
    if "backup" in kwargs:
        backup_config = kwargs["backup"]

    for epoch in range(start_epoch, epochs):
        ## train for one epoch
        train_epoch(device=device, loader=train_loader, model=model,
                    criterion=criterion, optimizer=optimizer,
                    epoch=epoch, log_str=train_log_str)

        ## evaluate on validation set
        prec1 = validate(device=device, loader=val_loader, model=model,
                         criterion=criterion, epoch=epoch,
                         log_str=val_log_str)

        # remember best prec@1 
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1) 
        print("Best Prec@1: %.3f\n" % (best_prec1))

        # save checkpoint
        if backup_config is not None:
            ckpt_dir = backup_config["dir_path"]
            pth_name = backup_config["pth_name"]
            os.makedirs(ckpt_dir, exist_ok=True)
            utils.save_checkpoint(state={
                                    "epoch": epoch,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "best_prec1": best_prec1
                                    },
                                is_best=is_best,
                                path=ckpt_dir,
                                prefix=pth_name)
