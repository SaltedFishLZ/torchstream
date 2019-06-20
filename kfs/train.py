"""
"""
import os
import gc
import time
import json
import argparse

import torch
import torchstream

import cfgs
import utils

import kfs

val_log_str = "Validation:[{:4d}/{:4d}],  " + \
              "BatchTime:{batch_time.val:6.2f}({batch_time.avg:6.2f}),  " + \
              "DataTime:{data_time.val:6.2f}({data_time.avg:6.2f}),  " + \
              "Loss:{loss_meter.val:7.3f}({loss_meter.avg:7.3f}),  " + \
              "Prec@1:{top1_meter.val:7.3f}({top1_meter.avg:7.3f}),  " + \
              "Prec@5:{top5_meter.val:7.3f}({top5_meter.avg:7.3f})"

def validate(device, loader, model, criterion,
             log_str=val_log_str, log_interval=20, **kwargs):
    
    batch_time = utils.Meter()
    data_time = utils.Meter()
    loss_meter = utils.Meter()
    top1_meter = utils.Meter()
    top5_meter = utils.Meter()

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


    print("Results:\n" + \
          "Prec@1 {top1_meter.avg:5.3f} Prec@5 {top5_meter.avg:5.3f} Loss {loss_meter.avg:5.3f}"
          .format(top1_meter=top1_meter, top5_meter=top5_meter, loss_meter=loss_meter))

    return top1_meter.avg

train_log_str = "Epoch:[{:3d}][{:4d}/{:4d}],  lr:{lr:5.5f},  " + \
                "BatchTime:{batch_time.val:6.2f}({batch_time.avg:6.2f}),  " + \
                "DataTime:{data_time.val:6.2f}({data_time.avg:6.2f}),  " + \
                "Loss:{loss_meter.val:7.3f}({loss_meter.avg:7.3f}),  " + \
                "Prec@1:{top1_meter.val:7.3f}({top1_meter.avg:7.3f}),  " + \
                "Prec@5:{top5_meter.val:7.3f}({top5_meter.avg:7.3f})"


def train(device, loader, model, criterion, optimizer, epoch,
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

        print(output.size(), target.size())
        # ## calculate accuracy
        # accuracy = metric(output.data, target)
        # prec1 = accuracy[1]
        # prec5 = accuracy[5]

        ## update statistics
        loss_meter.update(loss, input.size(0))
        # top1_meter.update(prec1, input.size(0))
        # top5_meter.update(prec5, input.size(0))

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





def main(args):

    ## args, configs -> configs
    configs = {}
    with open(args.config, "r") as json_config:
        configs = json.load(json_config)
    if args.gpus is not None:
        configs["gpus"] = args.gpus
    else:
        configs["gpus"] = list(range(torch.cuda.device_count()))
    configs["train"]["start_epoch"] = 0

    device = torch.device("cuda:0")
    best_prec1 = 0


    # -------------------------------------------------------- #
    #          Construct Datasets & Dataloaders                #
    # -------------------------------------------------------- #

    train_transforms = []
    for _t in configs["train_transforms"]:
        train_transforms.append(cfgs.config2transform(_t))
    train_transform = torchstream.transforms.Compose(
        transforms=train_transforms
        )

    configs["train_dataset"]["argv"]["transform"] = train_transform
    train_dataset = cfgs.config2dataset(configs["train_dataset"])

    configs["train_loader"]["dataset"] = train_dataset
    train_loader = cfgs.config2dataloader(configs["train_loader"])


    val_transforms = []
    for _t in configs["val_transforms"]:
        val_transforms.append(cfgs.config2transform(_t))
    val_transform = torchstream.transforms.Compose(
        transforms=val_transforms
        )

    configs["val_dataset"]["argv"]["transform"] = val_transform
    val_dataset = cfgs.config2dataset(configs["val_dataset"])

    configs["val_loader"]["dataset"] = val_dataset
    val_loader = cfgs.config2dataloader(configs["val_loader"])



    # -------------------------------------------------------- #
    #             Construct Network & Optimizer                #
    # -------------------------------------------------------- #

    model = kfs.Wrapper(**configs["model"]["argv"])
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=configs["gpus"])

    configs["optimizer"]["argv"]["params"] = model.parameters()
    optimizer = cfgs.config2optimizer(configs["optimizer"])

    lr_scheduler = cfgs.config2lrscheduler(optimizer, configs["lr_scheduler"])

    criterion = cfgs.config2criterion(configs["criterion"])
    criterion.to(device)



    # -------------------------------------------------------- #
    #            Resume / Finetune from Checkpoint             #
    # -------------------------------------------------------- #

    if "resume" in configs["train"]:
        resume_config = configs["train"]["resume"]
        checkpoint = utils.load_checkpoint(**resume_config)
        
        if checkpoint is not None:
            best_prec1 = checkpoint["best_prec1"]
            configs["train"]["start_epoch"] = start_epoch = checkpoint["epoch"] + 1
            model_state_dict = checkpoint["model_state_dict"]
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
            lr_scheduler_state_dict = checkpoint["lr_scheduler_state_dict"]

            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
            lr_scheduler.load_state_dict(lr_scheduler_state_dict)
            print("Resume from epoch [{}], best prec1 [{}]".format(start_epoch - 1, best_prec1))
        else:
            print("failed to load checkpoint")

    if "classifier" in configs["train"]:
        classifier_config = configs["train"]["classifier"]
        if "pretrained" in classifier_config:
            checkpoint = utils.load_checkpoint(**(classifier_config["pretrained"]))
            # load state_dict
            classifier_state_dict = checkpoint["model_state_dict"]
            old_keys = list(classifier_state_dict.keys())
            for old_key in old_keys:
                new_key = old_key.replace("module.", "")
                val = classifier_state_dict[old_key]
                del classifier_state_dict[old_key]
                classifier_state_dict[new_key] = val
            model.module.classifier.load_state_dict(classifier_state_dict)
        if "freeze" in classifier_config:
            if classifier_config["freeze"]:
                print("Freezing Classifier...")
                #  freeze classifier
                model.module.freeze_classifier()
                for m in model.module.classifier.modules():
                    if m.training:
                        print(m)

    # -------------------------------------------------------- #
    #                       Echo Config                        #
    # -------------------------------------------------------- #
    print(optimizer)
    print(lr_scheduler)


    # -------------------------------------------------------- #
    #                       Main Loop                          #
    # -------------------------------------------------------- #

    backup_config = None
    if "backup" in configs["train"]:
        backup_config = configs["train"]["backup"]

    for epoch in range(configs["train"]["start_epoch"],
                       configs["train"]["epochs"]):
        ## train for one epoch
        train(device=device, loader=train_loader, model=model,
              criterion=criterion, optimizer=optimizer,
              epoch=epoch)

        lr_scheduler.step()

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
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "best_prec1": best_prec1
                }
            utils.save_checkpoint(checkpoint=checkpoint,
                                  is_best=is_best,
                                  dir_path=dir_path,
                                  pth_name=pth_name)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Video Recognition Template")
    # configuration file
    parser.add_argument("config", type=str,
                        help="path to configuration file")
    parser.add_argument('--gpus', nargs='+', type=int, default=None)

    args = parser.parse_args()

    main(args)
