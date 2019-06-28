import os
import sys
import time
import json
import pickle
import shutil
import argparse

import torch
import torchstream
import numpy as np
import matplotlib.pyplot as plt

import cfgs
import utils
import kfs

test_log_str = "Testing:[{:4d}/{:4d}]  " + \
               "BatchTime:{batch_time.val:6.2f}({batch_time.avg:6.2f}),  " + \
               "DataTime:{data_time.val:6.2f}({data_time.avg:6.2f}),  " + \
               "Loss:{loss_meter.val:7.3f}({loss_meter.avg:7.3f}),  " + \
               "Prec@1:{top1_meter.val:7.3f}({top1_meter.avg:7.3f}),  " + \
               "Prec@5:{top5_meter.val:7.3f}({top5_meter.avg:7.3f})"

def test(device, loader, model, criterion, 
         log_str=test_log_str, log_interval=20, **kwargs):

    datapoint_num = 0
    # top1 error datapoint list
    top1_error_datapoints = []
    # top5 error datapoint list
    top5_error_datapoints = []

    batch_time = utils.Meter()
    data_time = utils.Meter()
    loss_meter = utils.Meter()
    top1_meter = utils.Meter()
    top5_meter = utils.Meter()

    metric = utils.ClassifyAccuracy(topk=(1, 5))

    model.eval()

    end = time.time()

    all_predicts = None
    all_targets = None

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            target = target.to(device)

            ## measure extra data loading time
            data_time.update(time.time() - end)

            output = model(input)
            loss = criterion(output, target)

            ## get predicts directly
            predict = utils.output2pred(output.data, maxk=5)
            correct = utils.classify_corrects(predict, target)
            for n in range(correct.size(1)):
                if correct[0][n].item() == 0:
                    top1_error_datapoints.append(datapoint_num + n)
                if correct[1][n].item() == 0:
                    top5_error_datapoints.append(datapoint_num + n)
            datapoint_num += correct.size(1)

            accuracy = metric(output.data, target)
            prec1 = accuracy[1]
            prec5 = accuracy[5]

            loss_meter.update(loss, input.size(0))
            top1_meter.update(prec1, input.size(0))
            top5_meter.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if all_predicts is None:
                all_predicts = predict[0]
                all_targets = target
            else:
                all_predicts = torch.cat((all_predicts, predict[0]))
                all_targets = torch.cat((all_targets, target))

            if i % log_interval == 0:
                print(log_str.format(i, len(loader),
                                     batch_time=batch_time,
                                     data_time=data_time,
                                     loss_meter=loss_meter,
                                     top1_meter=top1_meter,
                                     top5_meter=top5_meter))



    print("Results:\n"
          "Prec@1 {top1_meter.avg:5.3f}"
          "Prec@5 {top5_meter.avg:5.3f}"
          "Loss {loss_meter.avg:5.3f}"
          .format(top1_meter=top1_meter,
                  top5_meter=top5_meter,
                  loss_meter=loss_meter))

    # print("Top-5 Error List")
    # print(top5_error_datapoints)

    all_predicts = all_predicts.cpu()
    all_targets = all_targets.cpu()

    all_predicts = np.array(all_predicts)
    all_targets = np.array(all_targets)
    # print(all_predicts == all_targets)

    cm = utils.confusion_matrix(all_predicts, all_targets, normalize=True)

    # np.set_printoptions(threshold=sys.maxsize)
    # print(cm)

    with open("confusion_matrix.pkl", "wb") as f:
        pickle.dump(cm, f)
    # utils.plot_confusion_matrix(cm)
    # plt.show()

    return top1_error_datapoints


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
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=configs["gpus"])

    # load model
    checkpoint = torch.load(args.weights)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)

    criterion = cfgs.config2criterion(configs["criterion"])
    criterion.to(device)

    top5_error_datapoints = test(device, test_loader, model, criterion)
    lens = []
    # print("Top-5 Error Num", len(top5_error_datapoints))
    # for _i in top5_error_datapoints:
    #     print(test_dataset.datapoints[_i])
    #     print("Video Shape", np.array(test_dataset.samples[_i]).shape)
    #     # lens.append(np.array(test_dataset.samples[_i]).shape[0])

    # nphist = np.histogram(lens, bins=20)
    # print("NumPy Hist")
    # print("Density\n", nphist[0])
    # print("Bins\n", nphist[1])

    # plt.hist(lens, density=True, bins=20)
    # plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Video Recognition Template")
    # configuration file
    parser.add_argument("config", type=str,
                        help="path to configuration file")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--gpus", nargs='+', type=int, default=None)

    args = parser.parse_args()

    main(args)
