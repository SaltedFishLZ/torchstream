"""

Runtime configurations are specified in args, usually can be overriden by ENV.
Training hyper-parameters are specified in JSON files.

"""
import time
import json
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed as datadist
import torchstream

import cfgs
import utils

best_prec1 = 0

parser = argparse.ArgumentParser(description="Template Training Script")
parser.add_argument("config", type=str,
                    help="path to configuration file")
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--nodes', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='master node url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

# default val log
val_log_str = "Validation:[{:4d}/{:4d}],  " + \
              "BatchTime:{batch_time.val:6.2f}({batch_time.avg:6.2f}),  " + \
              "DataTime:{data_time.val:6.2f}({data_time.avg:6.2f}),  " + \
              "Loss:{loss_meter.val:7.3f}({loss_meter.avg:7.3f}),  " + \
              "Prec@1:{top1_meter.val:7.3f}({top1_meter.avg:7.3f}),  " + \
              "Prec@5:{top5_meter.val:7.3f}({top5_meter.avg:7.3f})"


def validate(gid, loader, model, criterion, shown_count=False,
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
            input = input.cuda(gid)
            target = target.cuda(gid)

            # measure extra data loading time
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
    if gid == 0:
        print("Results:\n"
              "Prec@1 {top1_meter.avg:5.3f} "
              "Prec@5 {top5_meter.avg:5.3f} "
              "Loss {loss_meter.avg:5.3f}"
              .format(top1_meter=top1_meter,
                      top5_meter=top5_meter,
                      loss_meter=loss_meter))

    if shown_count:
        return (top1_meter.avg, top1_meter.count)
    else:
        return top1_meter.avg


train_log_str = "Epoch:[{:3d}][{:4d}/{:4d}],  lr:{lr:5.5f},  " + \
                "BatchTime:{batch_time.val:6.2f}({batch_time.avg:6.2f}),  " + \
                "DataTime:{data_time.val:6.2f}({data_time.avg:6.2f}),  " + \
                "Loss:{loss_meter.val:7.3f}({loss_meter.avg:7.3f}),  " + \
                "Prec@1:{top1_meter.val:7.3f}({top1_meter.avg:7.3f}),  " + \
                "Prec@5:{top5_meter.val:7.3f}({top5_meter.avg:7.3f})"


def train(gid, loader, model, criterion,
          optimizer, lr_scheduler, epoch,
          log_str=train_log_str, log_interval=20,
          **kwargs):

    batch_time = utils.Meter()
    data_time = utils.Meter()
    loss_meter = utils.Meter()
    top1_meter = utils.Meter()
    top5_meter = utils.Meter()

    metric = utils.ClassifyAccuracy(topk=(1, 5))

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(loader):
        input = input.cuda(gid)
        target = target.cuda(gid)

        # measure extra data loading time
        data_time.update(time.time() - end)

        # forward
        output = model(input)
        loss = criterion(output, target)

        # calculate accuracy
        accuracy = metric(output.data.cpu(), target.cpu())
        prec1 = accuracy[1]
        prec5 = accuracy[5]

        # update statistics
        loss_meter.update(loss.data.cpu(), input.size(0))
        top1_meter.update(prec1, input.size(0))
        top5_meter.update(prec5, input.size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time on GPU
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

    # schedule lr after each epoch, not each batch!
    lr_scheduler.step()


def worker(pid, ngpus_per_node, args):
    """
    Note:
    Until platform setting, everything runs on CPU side.
    """

    # -------------------------------------------------------- #
    #                     Initialization                       s#
    # -------------------------------------------------------- #
    configs = {}
    with open(args.config, "r") as json_config:
        configs = json.load(json_config)

    args.gid = pid
    if pid is not None:
        torch.cuda.set_device(args.gid)
        print("Proc [{:2d}] Uses GPU [{:2d}]".format(pid, args.gid))

    if args.distributed:
        args.rank = args.rank * ngpus_per_node + args.gid
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    global best_prec1
    start_epoch = 0
    checkpoint = None

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

    # TODO: integrate into configuration?
    if args.distributed:
        train_sampler = datadist.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.distributed:
        configs["train_loader"]["batch_size"] = \
            int(configs["train_loader"]["batch_size"] / ngpus_per_node)
        configs["train_loader"]["num_workers"] = \
            int(configs["train_loader"]["num_workers"] / ngpus_per_node)
        # turn off the shuffle option outside, set shuffule in sampler
        configs["train_loader"]["shuffle"] = False

    configs["train_loader"]["dataset"] = train_dataset
    configs["train_loader"]["sampler"] = train_sampler
    configs["train_loader"]["shuffle"] = configs["train_loader"]["shuffle"] and (train_sampler is None)
    train_loader = cfgs.config2dataloader(configs["train_loader"])

    val_transforms = []
    for _t in configs["val_transforms"]:
        val_transforms.append(cfgs.config2transform(_t))
    val_transform = torchstream.transforms.Compose(
        transforms=val_transforms
        )

    configs["val_dataset"]["argv"]["transform"] = val_transform
    val_dataset = cfgs.config2dataset(configs["val_dataset"])

    # val_sampler -> None, all GPU do the same validation

    # if args.distributed:
    #     val_sampler = datadist.DistributedSampler(val_dataset)
    # else:
    #     val_sampler = None

    # if args.distributed:
    #     configs["val_loader"]["batch_size"] = \
    #         int(configs["val_loader"]["batch_size"] / ngpus_per_node)
    #     configs["val_loader"]["num_workers"] = \
    #         int(configs["val_loader"]["num_workers"] / ngpus_per_node)

    configs["val_loader"]["dataset"] = val_dataset
    # configs["val_loader"]["sampler"] = val_sampler
    val_loader = cfgs.config2dataloader(configs["val_loader"])

    # -------------------------------------------------------- #
    #                 Construct Neural Network                 #
    # -------------------------------------------------------- #

    model = cfgs.config2model(configs["model"])

    # load checkpoint
    if "resume" in configs["train"]:
        # NOTE: the 1st place to load checkpoint
        resume_config = configs["train"]["resume"]
        checkpoint = utils.load_checkpoint(**resume_config)
        if checkpoint is None:
            print("Load Checkpoint Failed")
        if checkpoint is not None:
            # check checkpoint device mapping
            model_state_dict = checkpoint["model_state_dict"]
            for _k in model_state_dict:
                _v = model_state_dict[_k]
                expected_device = "cuda:{}".format(args.gid)
                if expected_device != _v.device and "cuda" in str(_v.device):
                    raise ValueError("Device Mismatch [{}]: {} -> {}".format(
                        _k, expected_device, _v.device
                    ))
            print("Loading Checkpoint...")
            model.load_state_dict(model_state_dict)
    # ignore finetune if there is a checkpoint
    if (checkpoint is None) and ("finetune" in configs["train"]):
        finetune_config = configs["train"]["finetune"]
        checkpoint = utils.load_checkpoint(**finetune_config)
        if checkpoint is None:
            raise ValueError("Load Finetune Model Failed")
        # TODO: move load finetune model into model's method
        # not all models replace FCs only
        model_state_dict = checkpoint["model_state_dict"]
        for key in model_state_dict:
            if "fc" in key:
                # use FC from new network
                print("Replacing ", key)
                model_state_dict[key] = model.state_dict()[key]
        model.load_state_dict(model_state_dict)
        # set to None to prevent loading other states
        checkpoint = None

    # move to device
    model = model.cuda(args.gid)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gid],
            find_unused_parameters=True
        )
    else:
        model = torch.nn.DataParallel(model)

    # -------------------------------------------------------- #
    #            Construct Optimizer, Scheduler etc            #
    # -------------------------------------------------------- #
    print("Setting Optimizer & Lr Scheduler...")

    if configs["optimizer"]["argv"]["params"] == "model_specified":
        print("Use Model Specified Training Policies")
        configs["optimizer"]["argv"]["params"] = \
            model.module.get_optim_policies()
    else:
        print("Train All Parameters")
        configs["optimizer"]["argv"]["params"] = model.parameters()
    optimizer = cfgs.config2optimizer(configs["optimizer"])
    lr_scheduler = cfgs.config2lrscheduler(optimizer, configs["lr_scheduler"])

    if "resume" in configs["train"]:
        if checkpoint is not None:
            best_prec1 = checkpoint["best_prec1"]
            start_epoch = checkpoint["epoch"] + 1
            model_state_dict = checkpoint["model_state_dict"]
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
            lr_scheduler_state_dict = checkpoint["lr_scheduler_state_dict"]

            optimizer.load_state_dict(optimizer_state_dict)
            lr_scheduler.load_state_dict(lr_scheduler_state_dict)
            print("Resume from epoch [{}], best prec1 [{}]".
                  format(start_epoch - 1, best_prec1))

    criterion = cfgs.config2criterion(configs["criterion"])
    criterion = criterion.cuda(args.gid)

    # -------------------------------------------------------- #
    #                       Main Loop                          #
    # -------------------------------------------------------- #

    backup_config = None
    if "backup" in configs["train"]:
        backup_config = configs["train"]["backup"]
    epochs = configs["train"]["epochs"]

    print("Training Begins")

    for epoch in range(start_epoch, epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(gid=args.gid,
              loader=train_loader,
              model=model, criterion=criterion,
              optimizer=optimizer, lr_scheduler=lr_scheduler,
              epoch=epoch)

        # evaluate on validation set
        if args.gid == 0:
            prec1 = validate(gid=args.gid,
                             loader=val_loader,
                             model=model, criterion=criterion,
                             epoch=epoch)
            # remember best prec@1
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            print("Best Prec@1: %.3f\n" % (best_prec1))

            # save checkpoint
            if backup_config is not None:
                dir_path = backup_config["dir_path"]
                pth_name = backup_config["pth_name"]

                model_state_dict = model.state_dict()
                optimizer_state_dict = optimizer.state_dict()
                lr_scheduler_state_dict = lr_scheduler.state_dict()

                # remove prefixes in (distributed) data parallel wrapper
                utils.checkpoint.remove_prefix_in_keys(model_state_dict)

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_prec1": best_prec1
                    }
                utils.save_checkpoint(checkpoint=checkpoint,
                                      is_best=is_best,
                                      dir_path=dir_path,
                                      pth_name=pth_name)


def main():
    args = parser.parse_args()

    # When using multiple nodes, automatically set distributed
    if args.nodes > 1:
        args.distributed = True

    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        # We have ngpus_per_node processes per node
        args.world_size = ngpus_per_node * args.nodes
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # worker process function
        mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call worker function
        worker(0, ngpus_per_node, args)


if __name__ == '__main__':
    main()
