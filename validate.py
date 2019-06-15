"""
"""
import time

import torch

import utils

val_log_str = "Validation: [{:4d}/{:4d}]\t" + \
              "BatchTime {batch_time.val:5.3f} ({batch_time.avg:5.3f})\t" + \
              "DataTime {data_time.val:5.3f} ({data_time.avg:5.3f})\t" + \
              "Loss {loss.val:5.3f} ({loss.avg:5.3f})\t" + \
              "Prec@1 {top1.val:5.3f} ({top1.avg:5.3f})\t" + \
              "Prec@5 {top5.val:5.3f} ({top5.avg:5.3f})"


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


    print("Validating Results:\n" + \
          "Prec@1 {top1.avg:5.3f} Prec@5 {top5.avg:5.3f} Loss {loss.avg:5.3f}"
          .format(top1=top1, top5=top5, loss=losses))

    return top1.avg
