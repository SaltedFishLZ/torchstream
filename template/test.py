import os, json, time, pickle
import argparse

import torch
from torch.utils.data import RandomSampler
import torchstream
import numpy as np
import matplotlib.pyplot as plt

import cfgs
import utils

test_log_str = "Testing:[{:4d}/{:4d}]  " + \
               "BatchTime:{batch_time.val:6.2f}({batch_time.avg:6.2f}),  " + \
               "DataTime:{data_time.val:6.2f}({data_time.avg:6.2f}),  " + \
               "Loss:{loss_meter.val:7.3f}({loss_meter.avg:7.3f}),  " + \
               "Prec@1:{top1_meter.val:7.3f}({top1_meter.avg:7.3f}),  " + \
               "Prec@5:{top5_meter.val:7.3f}({top5_meter.avg:7.3f})"


def test(device, loader, model, criterion,
         log_str=test_log_str, log_interval=20,
         trace_dir=None, dump_logit=False, dump_prob=False,
         dump_pred=False, dump_correct=False,
         **kwargs):

    if ((dump_logit or dump_prob or dump_pred or dump_correct)
            and isinstance(loader.sampler, RandomSampler)):
        raise ValueError("DataLoader cannot be shuffled when dumping traces!")
    if ((dump_logit or dump_prob or dump_pred or dump_correct)
            and (trace_dir is None)):
        raise ValueError("Must Specify Trace Directory!")

    batch_time = utils.Meter()
    data_time = utils.Meter()
    loss_meter = utils.Meter()
    top1_meter = utils.Meter()
    top5_meter = utils.Meter()

    metric = utils.ClassifyAccuracy(topk=(1, 5))

    model.eval()

    end = time.time()

    trace_logit = None
    trace_prob = None
    trace_pred = None
    trace_correct = None

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            target = target.to(device)

            # measure extra data loading time
            data_time.update(time.time() - end)

            # get result
            output = model(input)

            # dump trace
            if dump_logit:
                if trace_logit is None:
                    trace_logit = output.cpu()
                else:
                    trace_logit = torch.cat((trace_logit, output.cpu()))

            prob = torch.nn.functional.softmax(output).cpu()
            if dump_prob:
                if trace_prob is None:
                    trace_prob = prob.cpu()
                else:
                    trace_prob = torch.cat((trace_prob, prob.cpu()))

            predict = utils.metrics.output2pred(output, maxk=5)
            if dump_pred:
                if trace_pred is None:
                    trace_pred = predict.cpu()
                else:
                    trace_pred = torch.cat((trace_pred, predict.cpu()), dim=1)

            correct = utils.metrics.classify_corrects(predict, target)
            if dump_correct:
                if trace_correct is None:
                    trace_correct = correct.cpu()
                else:
                    trace_correct = torch.cat((trace_correct, correct.cpu()), dim=1)

            # measure loss & accuracy
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

    # save trace to files

    if dump_logit:
        file_path = os.path.join(trace_dir, "trace_logit.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(trace_logit, f)

    if dump_prob:
        file_path = os.path.join(trace_dir, "trace_prob.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(trace_prob, f)

    if dump_pred:
        file_path = os.path.join(trace_dir, "trace_pred.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(trace_pred, f)

    if dump_correct:
        file_path = os.path.join(trace_dir, "trace_correct.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(trace_correct, f)

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

    model = cfgs.config2model(configs["model"])

    # load model
    checkpoint = torch.load(args.weights)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=configs["gpus"])

    criterion = cfgs.config2criterion(configs["criterion"])
    criterion.to(device)

    test(device, test_loader, model, criterion,
         trace_dir="test",
         dump_logit=args.dump_logit,
         dump_prob=args.dump_prob,
         dump_pred=args.dump_pred,
         dump_correct=args.dump_correct
         )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Template Testing Script")

    # configuration file
    parser.add_argument("config", type=str,
                        help="path to configuration file")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--gpus", nargs='+', type=int, default=None)

    parser.add_argument("--trace_dir", type=str, default=None)
    parser.add_argument("--dump_logit", action="store_true")
    parser.add_argument("--dump_prob", action="store_true")
    parser.add_argument("--dump_pred", action="store_true")
    parser.add_argument("--dump_correct", action="store_true")

    args = parser.parse_args()

    main(args)
