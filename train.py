"""
"""
import os
import time
import json
import shutil

import torch

import opts
import cfgs
import utils

train_log_str = "Epoch: [{:3d}][{:4d}/{:4d}], lr: {lr:.5f}\t" + \
                "Calc(s) {batch_time.val:.3f} ({batch_time.avg:.3f})\t" + \
                "Data(s) {data_time.val:.3f} ({data_time.avg:.3f})\t" + \
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


val_log_str = "Test: [{:4d}/{:4d}]\t" + \
              "Calc(s) {batch_time.val:.3f} ({batch_time.avg:.3f})\t" + \
              "Data(s) {data_time.val:.3f} ({data_time.avg:.3f})\t" + \
              "Loss {loss.val:.4f} ({loss.avg:.4f})\t" + \
              "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t" + \
              "Prec@5 {top5.val:.3f} ({top5.avg:.3f})"


def validate(device, loader, model, criterion,
             log_str, log_interval=20, **kwargs):
    
    batch_time = utils.Meter()
    data_time = utils.Meter()
    losses = utils.Meter()
    top1 = utils.Meter()
    top5 = utils.Meter()

    metric = utils.ClassifyAccuracy(topk=(1, 5))

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            target = target.to(device)

            ## measure extra data loading time
            data_time.update(time.time() - end)

            output = model(input)
            loss = criterion(output, target)

            accuracy = metric(output.data, target)
            prec1 = accuracy[1]
            prec5 = accuracy[5]

            losses.update(loss, input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_interval == 0:
                print(log_str.format(i, len(loader),
                                     batch_time=batch_time,
                                     data_time=data_time,
                                     loss=losses, top1=top1, top5=top5))


    print("Testing Results:\n" + \
          "Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}"
          .format(top1=top1, top5=top5, loss=losses))

    return top1.avg



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


    best_prec1 = 0



    for epoch in range(configs["train"]["start_epoch"],
                       configs["train"]["epochs"]):
        
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
        output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
        print(output_best)

        # save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        utils.save_checkpoint(state={
                                    "epoch": epoch,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                    "best_prec1": best_prec1
                                    },
                              is_best=is_best,
                              path="checkpoints",
                              prefix="template")


if __name__ == "__main__":
    args = opts.parser.parse_args()

    main(args)
