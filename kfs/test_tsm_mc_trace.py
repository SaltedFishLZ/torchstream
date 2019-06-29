import os
import time
import json
import pickle
import shutil
import argparse

import torch
import torchstream

import cfgs
import utils


test_log_str = "Testing:[{:4d}/{:4d}]  " + \
               "BatchTime:{batch_time.val:6.2f}({batch_time.avg:6.2f}),  " + \
               "DataTime:{data_time.val:6.2f}({data_time.avg:6.2f}),  " + \
               "Loss:{loss_meter.val:7.3f}({loss_meter.avg:7.3f}),  " + \
               "Prec@1:{top1_meter.val:7.3f}({top1_meter.avg:7.3f}),  " + \
               "Prec@5:{top5_meter.val:7.3f}({top5_meter.avg:7.3f})"

def test_mc_with_trace(device, loader, model, criterion, chances=20,
            log_str=test_log_str, log_interval=20, **kwargs):


    mc_acc = utils.MultiChanceClassifyAccuracy(topk=(1, 5))

    model.eval()

    result = {
            "multi-chance": {
                "top1": [],
                "top5": []
            },
            "single-chance": {
                "top1": [],
                "top5": []
            }
        }

    with torch.no_grad():

        for c in range(chances):

            print("#" * 40)
            print("[{:5d}] Chances".format(c))
            print("#" * 40)

            batch_time = utils.Meter()
            data_time = utils.Meter()
            loss_meter = utils.Meter()
            top1_meter = utils.Meter()
            top5_meter = utils.Meter()

            metric = utils.ClassifyAccuracy(topk=(1, 5))

            all_output = None
            all_target = None
            all_index = None
            all_length = None
            all_correct = None
            all_loss = None

            end = time.time()
            for i, ((input, index, length), target) in enumerate(loader):

                input = input.to(device)
                target = target.to(device)

                # print(length.size())

                ## measure extra data loading time
                data_time.update(time.time() - end)

                output = model(input)
                loss = criterion(output, target)

                accuracy = metric(output.data, target)
                prec1 = accuracy[1]
                prec5 = accuracy[5]

                loss_meter.update(loss.mean(), input.size(0))
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
                ## 
                pred = utils.output2pred(output.data, 1)
                correct = utils.classify_corrects(pred, target)[0]

                if all_output is None:
                    all_output = output
                    all_target = target
                    all_index = index
                    all_length = length
                    all_correct = correct
                    all_loss = loss
                else:
                    all_output = torch.cat((all_output, output))
                    all_target = torch.cat((all_target, target))
                    all_index = torch.cat((all_index, index))
                    all_length = torch.cat((all_length, length))
                    all_correct = torch.cat((all_correct, correct))
                    all_loss = torch.cat((all_loss, loss))

            # debug
            # print(all_index.size())
            # print(all_index)
            trace_dir = "new-mc-trace/chance{}".format(c)
            os.makedirs(trace_dir)
            f = open(os.path.join(trace_dir, "index.pkl"), "wb")
            pickle.dump(all_index, f)
            f.close()
            f = open(os.path.join(trace_dir, "length.pkl"), "wb")
            pickle.dump(all_length, f)
            f.close()
            f = open(os.path.join(trace_dir, "correct.pkl"), "wb")
            pickle.dump(all_correct, f)
            f.close()
            f = open(os.path.join(trace_dir, "loss.pkl"), "wb")
            pickle.dump(all_loss, f)
            f.close()

            print("This Chance:\n" + \
                "Prec@1 {top1_meter.avg:5.3f} Prec@5 {top5_meter.avg:5.3f}"
                .format(top1_meter=top1_meter, top5_meter=top5_meter))

            result["single-chance"]["top1"].append(top1_meter.avg)
            result["single-chance"]["top5"].append(top5_meter.avg)

            multichance_accuracy = mc_acc(all_output, all_target)
            mc_top1 = multichance_accuracy[1]
            mc_top5 = multichance_accuracy[5]
            print("All Chances:\nPrec@1 {:5.3f} Prec@5 {:5.3f}"
                  .format(mc_top1, mc_top5))

            result["multi-chance"]["top1"].append(mc_top1)
            result["multi-chance"]["top5"].append(mc_top5)

    return result

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

    import transforms

    test_transforms.append(transforms.CenterSegment(size=16))
    test_transforms.append(transforms.RandomFrameSampler(size=8, output_index=True, output_length=True))
    test_transforms.append(
            transforms.Fork(transforms=[
                torchstream.transforms.Compose(
                    [
                        torchstream.transforms.Resize(size=256),
                        torchstream.transforms.CenterCrop(size=[224, 224]),
                        torchstream.transforms.ToTensor(),
                        torchstream.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ]
                ),
                lambda x: torch.Tensor(x),
                lambda y: torch.Tensor(y)
            ])
    )


    configs["test_dataset"]["argv"]["transform"] = torchstream.transforms.Compose(test_transforms)
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
    print("Model Best Prec1: ", checkpoint["best_prec1"])

    # criterion = cfgs.config2criterion(configs["criterion"])
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    criterion.to(device)

    result = test_mc_with_trace(device, test_loader, model, criterion, args.chances)
    with open(args.output, "wb") as fout:
        pickle.dump(result, fout)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="KFS MC Test")
    # configuration file
    parser.add_argument("config", type=str,
                        help="path to configuration file")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--chances", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gpus", nargs='+', type=int, default=None)
    
    args = parser.parse_args()

    assert args.output is not None, ValueError("Must Specify Output File")

    main(args)
