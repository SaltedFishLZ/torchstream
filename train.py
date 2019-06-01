"""
"""
import time

import torch

from utils import Meter, accuracy


def train(loader, model, criterion, optimizer, epoch, print_interval=20,
          **kwargs):
    
    batch_time = Meter()
    data_time = Meter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # N C T H W -> N T C H W
        input = input.permute(0, 2, 1, 3, 4).contiguous()
        # merge time to batch
        N, T, C, H, W = input.size()
        input = input.view(N*T, C, H, W)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        ## forward
        output = model(input_var)

        loss = criterion(output, target_var)

        ## measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        ## backward
        optimizer.zero_grad()

        loss.backward() 

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_interval == 0:
            print('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5, 
                      lr=optimizer.param_groups[-1]['lr']
                      )
                  )


def validate(loader, model, criterion, print_interval=20, **kwargs):
    batch_time = Meter()
    losses = Meter()
    top1 = Meter()
    top5 = Meter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            target = target.cuda()

            # N C T H W -> N T C H W
            input = input.permute(0, 2, 1, 3, 4).contiguous()
            # merge time to batch
            N, T, C, H, W = input.size()
            input = input.view(N*T, C, H, W)


            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))
                print(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)

    return top1.avg
    
