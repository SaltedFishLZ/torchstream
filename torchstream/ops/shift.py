"""
"""
__all__ = ["TemporalShift"]

import torch
import torch.nn as nn
import torch.nn.functional as F




class InplaceTemporalShiftFunction(torch.autograd.Function):
    """
    Shift `fold` channels
    Args
        x (Tensor): [N][T][C][H][W]
    """
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None



inplace_temporal_shift_function = InplaceTemporalShiftFunction.apply

def temporal_shift_function(x, fold):
    """
    Shift `fold` channels
    Args
        x (Tensor): [N][T][C][H][W]
    """
    out = torch.zeros_like(x)
    # shift left 
    out[:, :-1, :fold] = x[:, 1:, :fold]
    # shift right
    out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
    # not shift
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
    return out



class TemporalShift(nn.Module):
    """
    Args
        fold_div: (1/fold_div) frames will be shifted
    """
    def __init__(self, seg_num=3, fold_div=8, inplace=False, verbose=False):
        super(TemporalShift, self).__init__()
        self.seg_num = seg_num
        self.fold_div = fold_div
        self.inplace = inplace
        if verbose:
            if inplace:
                print('=> Using in-place shift...')
            print('=> Using fold div: {}'.format(self.fold_div))

    def __repr__(self):
        return self.__class__.__name__ + \
            "fold div: {}, inplace: {}".format(self.fold_div, self.inplace)

    def forward(self, x):
        ## reshape tensor
        nt, c, h, w = x.size()
        n = nt // self.seg_num
        t = self.seg_num
        fold = c // self.fold_div

        x = x.view(n, t, c, h, w)
        
        if self.inplace:
            x = inplace_temporal_shift_function(x, fold=fold)
        else:
            x = temporal_shift_function(x, fold=fold)
        
        ## reshape tensor
        x = x.view(nt, c, h, w)

        return x










if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    tsm1 = TemporalShift(seg_num=8, fold_div=8, inplace=False)
    tsm2 = TemporalShift(seg_num=8, fold_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224)
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224)
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5

    print('=> Testing GPU...')
    tsm1.cuda()
    tsm2.cuda()
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 3, 224, 224).cuda()
            y1 = tsm1(x)
            y2 = tsm2(x)
            assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
            x1.requires_grad_()
            x2 = x1.clone()
            y1 = tsm1(x1)
            y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            assert torch.norm(grad1 - grad2).item() < 1e-5
    print('Test passed.')
