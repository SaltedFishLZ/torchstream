"""
"""
import time

import torch

import utils

val_log_str = "Validation: [{:4d}/{:4d}]\t" + \
              "BatchTime {batch_time.val:6.2f} ({batch_time.avg:6.2f})\t" + \
              "DataTime {data_time.val:6.2f} ({data_time.avg:6.2f})\t" + \
              "Loss {loss_meter.val:6.3f} ({loss_meter.avg:6.3f})\t" + \
              "Prec@1 {top1_meter.val:6.3f} ({top1_meter.avg:6.3f})\t" + \
              "Prec@5 {top5_meter.val:6.3f} ({top5_meter.avg:6.3f})"

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


    print("Validating Results:\n" + \
          "Prec@1 {top1_meter.avg:5.3f} Prec@5 {top5_meter.avg:5.3f} Loss {loss_meter.avg:5.3f}"
          .format(top1_meter=top1_meter, top5_meter=top5_meter, loss_meter=loss_meter))

    return top1_meter.avg
