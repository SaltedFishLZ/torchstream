"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalInterpolationFunction(torch.autograd.Function):
    """Temporal Linear Interpolation
    """
    @staticmethod
    def forward(ctx, u, t):
        """
        Args
            u (Tensor): [N][Ti][C][H][W]
            t (Tensor): [N][To], values in [0, 1)
        """
        assert u.size(0) == t.size(0)
        N, Ti, C, H, W = u.size()
        N, To = t.size()

        # tl, tr, alpha: [N][To]
        tl = torch.floor(t * (Ti - 1)).long()
        tr = tl + 1
        tr = torch.where(tr < Ti, tr, (Ti - 1) * torch.ones_like(tr))

        alpha = t * (Ti - 1) - tl.float()

        # ul: left endpoint
        # ur: right endpoint
        ul = torch.zeros(N, To, C, H, W, device=u.device, requires_grad=False)
        ur = torch.zeros(N, To, C, H, W, device=u.device, requires_grad=False)
        for n in range(N):
            ul[n, :, :, :, :] = u[n, tl[n], :, :, :]
            ur[n, :, :, :, :] = u[n, tr[n], :, :, :]

        # save for backward
        ctx.Ti = Ti
        ctx.save_for_backward(ul, ur)

        # expand alpha to [N][To][C][H][W]
        alpha_e = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)\
            .expand(N, To, C, H, W)

        v = (1 - alpha_e) * ul + alpha_e * ur
        return v

    @staticmethod
    def backward(ctx, grad_output):

        # grad_output: [N][To][C][H][W]
        assert len(grad_output.size()) == 5, ValueError

        Ti = ctx.Ti
        ul, ur = ctx.saved_variables

        grad_u = grad_t = None
        if ctx.needs_input_grad[0]:
            # TODO
            raise NotImplementedError
        if ctx.needs_input_grad[1]:
            grad_t = (grad_output * (ur - ul) * (Ti - 1)).sum(dim=(2, 3, 4))

        return grad_u, grad_t


def temporal_interpolation_testbench(u, t):
    """
    """
    assert u.size(0) == t.size(0)
    N, Ti, C, H, W = u.size()
    N, To = t.size()

    # anchors: 0, 1, 3, ..., Ti-1
    anchors = torch.Tensor(range(Ti))

    # unnormalized input
    dist = torch.abs( ((Ti - 1) * t).unsqueeze(dim=-1).expand(N, To, Ti) - anchors.unsqueeze(dim=0).unsqueeze(dim=0).expand(N, To, Ti) )
    coef = torch.max(torch.zeros_like(dist), 1 - dist)

    # expand to [N][To][Ti][C][H][W]
    coef = coef.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(N, To, Ti, C, H, W)
    u = u.unsqueeze(dim=1).expand(N, To, Ti, C, H, W)
    v = (coef * u).sum(dim=2)

    return v


temporal_interpolation = TemporalInterpolationFunction.apply
# temporal_interpolation = temporal_interpolation_testbench

class TemporalInterpolationModule(nn.Module):
    """
    """
    def __init__(self, norm=True, mode="interval"):
        assert mode in ["time", "interval"], ValueError
        super(TemporalInterpolationModule, self).__init__()
        self.norm = norm
        self.mode = mode

    def __repr__(self):
        string = self.__class__.__name__
        string += " norm: {}, mode: {}".format(self.norm, self.mode)
        return string

    def forward(self, input, index):
        """
        Args:
            input [N][T][C][H][W]
            index [N][To + 1]
        """
        assert len(index.size()) == 2, ValueError("index [N][To + 1]")
        assert index.size(1) > 1, ValueError("too short")
        if self.mode == "interval":
            if self.norm:
                # normalize to [0, 1) and make sure sum = 1
                index = F.softmax(index, dim=1)
            index = index.cumsum(dim=1)
        else:
            assert ((index >= 0.0) & (index < 1.0)).prod().item() == 1, \
                ValueError
        # drop last
        index = index[:, :-1]
        return temporal_interpolation(input, index)


if __name__ == "__main__":
    N, T, C, H, W = 2, 10, 3, 100, 100

    t = torch.Tensor([
                      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
                      [0.0, 0.111, 1/9, 0.23, 0.2134, 5/9, 1.0],
                     ])
    t.requires_grad_(True)

    for i in range(20):
        u = torch.rand(N, T, C, H, W)
        
        v = temporal_interpolation(u, t)
        output = 1/2 * (v ** 2).sum()
        output.backward()
        grad = torch.Tensor(t.grad)

        v_tb = temporal_interpolation_testbench(u, t)
        output_tb = 1/2 * (v_tb ** 2).sum()
        output_tb.backward()
        grad_tb = torch.Tensor(t.grad)

        equal = (grad == grad_tb)
        passed = equal.prod().item()
        
        if not passed:
            print(equal)
    

    u = torch.rand(N, T, C, H, W)
    t = torch.Tensor([
                      [0.0, 0.12, 0.23, 0.34, 0.45, 0.56, 0.99],
                      [0.0, 1/9, 2/9, 3/9, 4/9, 5/9, 8/9],
                     ])
    t.requires_grad_(True)
    v = temporal_interpolation(u, t)
    output = v.sum()
    output.backward()
    grad = torch.Tensor(t.grad)
    print(t.grad)
    t = torch.Tensor([
                      [0.0, 1/9, 2/9, 3/9, 4/9, 5/9, 8/9],
                      [0.0, 0.12, 0.23, 0.34, 0.45, 0.56, 0.99],
                     ])
    t.requires_grad_(True)
    v = temporal_interpolation(u, t)
    output = v.sum()
    output.backward()
    grad = torch.Tensor(t.grad)
    print(t.grad)