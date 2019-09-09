"""
"""
__all__ = ["TemporalShift"]
import sys
import collections

import torch
import torch.nn as nn

if sys.version_info < (3, 3):
    Iterable = collections.Iterable
else:
    Iterable = collections.abc.Iterable


class InplaceTemporalShiftFunction(torch.autograd.Function):
    """
    Shift `fold` channels
    Args
        x (Tensor): [N][T][C][H][W]
    """
    @staticmethod
    def forward(ctx, input, fold, step=1):
        # not support higher order gradient
        # input = input.detach_()
        assert isinstance(step, int), TypeError
        ctx.fold_ = fold
        ctx.step_ = step
        n, t, c, h, w = input.size()
        assert (step >= 0 and step <= t), ValueError("Invalid Shift Step")

        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-step] = input.data[:, step:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, step:] = input.data[:, :-step, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        step = ctx.step_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, step:] = grad_output.data[:, :-step, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-step] = grad_output.data[:, step:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


inplace_temporal_shift_function = InplaceTemporalShiftFunction.apply


def temporal_shift_function(x, fold, steps=1):
    """
    Shift `fold` channels for each step size specified
    Args
        x (Tensor): [N][T][C][H][W]
    """
    n, t, c, h, w = x.size()
    assert isinstance(steps, (int, Iterable)), TypeError
    if isinstance(steps, int):
        steps = [steps]
    for step in steps:
        assert (step >= 0 and step <= t), ValueError("Invalid Shift Step")
    assert len(steps) * 2 * fold <= c, ValueError("Channel size too small")

    out = torch.zeros_like(x)

    shift_count = 0
    for step in steps:
        # shift left
        out[:, :-step, shift_count * fold: (shift_count + 1) * fold] \
            = x[:, step:, shift_count * fold: (shift_count + 1) * fold]
        shift_count += 1
        # shift right
        out[:, step:, shift_count * fold: (shift_count + 1) * fold] \
            = x[:, :-step, shift_count * fold: (shift_count + 1) * fold]
        shift_count += 1
    # no shift
    out[:, :, shift_count * fold:] = x[:, :, shift_count * fold:]

    return out


class TemporalShift(nn.Module):
    """
    Args
        fold_div: (1/fold_div) channels will be shifted
    """
    def __init__(self, seg_num=3, fold_div=8, shift_steps=1,
                 inplace=False, verbose=False):
        super(TemporalShift, self).__init__()
        self.seg_num = seg_num
        self.fold_div = fold_div
        self.shift_steps = shift_steps
        self.inplace = inplace
        if verbose:
            if inplace:
                print('=> Using in-place shift...')
            print('=> Using fold div: {}'.format(self.fold_div))

    def __repr__(self):
        string = self.__class__.__name__
        string += "(fold div: {}, inplace: {}, step: )"
        return string.format(self.fold_div, self.inplace, self.shift_step)

    def forward(self, x):
        # reshape tensor
        nt, c, h, w = x.size()
        n = nt // self.seg_num
        t = self.seg_num
        fold = c // self.fold_div

        # unfold to volume
        x = x.view(n, t, c, h, w)

        # call shift function
        if self.inplace:
            # x = inplace_temporal_shift_function(x, fold, self.shift_steps)
            raise NotImplementedError
        else:
            x = temporal_shift_function(x, fold, self.shift_steps)

        # reshape tensor
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
