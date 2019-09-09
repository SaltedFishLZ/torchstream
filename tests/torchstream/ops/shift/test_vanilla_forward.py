import torch
from torchstream.ops.shift import TemporalShift

if __name__ == '__main__':
    tsm1 = TemporalShift(seg_num=8, fold_div=8, inplace=False)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)

    print('=> Testing GPU...')
    tsm1.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)

    print('Test passed.')
