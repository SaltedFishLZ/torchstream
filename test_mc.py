import os
import time
import json
import shutil
import argparse

import torch

import cfgs
import utils


test_log_str = "Testing:[{:4d}/{:4d}]  " + \
               "BatchTime:{batch_time.val:6.2f}({batch_time.avg:6.2f}),  " + \
               "DataTime:{data_time.val:6.2f}({data_time.avg:6.2f}),  " + \
               "Loss:{loss_meter.val:7.3f}({loss_meter.avg:7.3f}),  " + \
               "Prec@1:{top1_meter.val:7.3f}({top1_meter.avg:7.3f}),  " + \
               "Prec@5:{top5_meter.val:7.3f}({top5_meter.avg:7.3f})"

def test_mc(device, loader, model, criterion, chances=20,
            log_str=test_log_str, log_interval=20, **kwargs):
    

    mc_acc = utils.MultiChanceClassifyAccuracy(topk=(1, 5))

    model.eval()

    

    with torch.no_grad():

        for c in range(chances):
            
            batch_time = utils.Meter()
            data_time = utils.Meter()
            loss_meter = utils.Meter()
            top1_meter = utils.Meter()
            top5_meter = utils.Meter()

            metric = utils.ClassifyAccuracy(topk=(1, 5))

            all_output = None
            all_target = None

            end = time.time()
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

                if all_output is None:
                    all_output = output
                    all_target = target
                else:
                    all_output = torch.cat((all_output, output))
                    all_target = torch.cat((all_target, target))

            print("This Chance:\n" + \
                "Prec@1 {top1_meter.avg:5.3f} Prec@5 {top5_meter.avg:5.3f}"
                .format(top1_meter=top1_meter, top5_meter=top5_meter))

            multichance_accuracy = mc_acc(all_output, all_target)
            mc_top1 = multichance_accuracy[1]
            mc_top5 = multichance_accuracy[5]
            print("[{:5d}] Chances:\n" + \
                "Prec@1 {:5.3f} Prec@5 {:5.3f}"
                .format(c, mc_top1, mc_top5))

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

    test_dataset = cfgs.config2dataset(configs["test_dataset"])


    configs["test_loader"]["dataset"] = test_dataset
    test_loader = cfgs.config2dataloader(configs["test_loader"])

    model = cfgs.config2model(configs["model"])
    model.to(device)

    # Add DataParallel Wrapper
    if args.gpus is not None:
        model = torch.nn.DataParallel(model, device_ids=args.gpus)
    else:
        model = torch.nn.DataParallel(model)
    
    # load model
    checkpoint = torch.load(args.weights)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)


    criterion = cfgs.config2criterion(configs["criterion"])
    criterion.to(device)


    test_mc(device, test_loader, model, criterion)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Video Recognition Template")
    # configuration file
    parser.add_argument("config", type=str,
                        help="path to configuration file")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--gpus", nargs='+', type=int, default=None)
    
    args = parser.parse_args()

    main(args)
