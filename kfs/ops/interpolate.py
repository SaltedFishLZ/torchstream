"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalInterpolationFunction(torch.autograd.Function):
    """
    Args:

    """
    @staticmethod
    def forward(ctx, u, t):
        """
        u [N][Ti][C][H][W]
        t [N][To], value belongs to [0, 1)
        """
        assert u.size(0) == t.size(0)
        N, Ti, C, H, W = u.size()
        N, To = t.size()

        # tl, tr, alpha: [N][To]
        tl = torch.floor(t * (Ti - 1)).long()
        tr = tl + 1
        alpha = t * (Ti - 1) - tl.float()

        ul = torch.zeros(N, To, C, H, W, device=u.device, requires_grad=False)
        ur = torch.zeros(N, To, C, H, W, device=u.device, requires_grad=False)
        for n in range(N):
            ul[n, :, :, :, :] = u[n, tl[n], :, :, :]
            ur[n, :, :, :, :] = u[n, tr[n], :, :, :]

        # ctx.Ti = Ti
        # ctx.tl = tl
        # ctx.tr = tr
        # ctx.ul = ul
        # ctx.ur = ur
        # ctx.alpha = alpha
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
        # tl = ctx.tl
        # tr = ctx.tr
        ul, ur = ctx.saved_variables

        grad_u = grad_t = None
        if ctx.needs_input_grad[0]:
            ## TODO
            raise NotImplementedError
        if ctx.needs_input_grad[1]:
            grad_t = (grad_output * (ur -ul) * (Ti - 1)).sum(dim=(2, 3, 4))

        return grad_u, grad_t

temporal_interpolation = TemporalInterpolationFunction.apply


class TemporalInterpolationModule(nn.Module):
    """
    """
    def __init__(self, norm=True, mode="interval"):
        assert mode in ["time", "interval"], ValueError
        super(TemporalInterpolationModule, self).__init__()
        self.norm = norm
        self.mode = mode

    def __repr__(self):
        return self.__class__.__name__ + " norm: {}, mode: {}".format(self.norm, self.mode)

    def forward(self, input, index):
        """
        index [N][To + 1]
        """
        assert len(index.size()) == 2, ValueError("index [N][To + 1]")
        assert index.size(1) > 1, ValueError("too short")
        if self.norm:
            index = F.softmax(index, dim=1)
        if self.mode == "interval":
            index = index.cumsum(dim=1)
        # drop last
        index = index[:, :-1]
        return temporal_interpolation(input, index)

    


if __name__ == "__main__":
    N, T, C, H, W = 2, 10, 3, 224, 224
    
    # u = torch.ones(N, T, C, H, W)
    u = torch.empty(N, T, C, H, W)
    for t in range(T):
        # u[0, t] = t
        u[0, t] = t **2
        u[1, t] = 2 * t **2

    t = torch.Tensor([[0.5, 0], [0.5, 0.9999999]])
    t.requires_grad_(True)
    v = temporal_interpolation(u, t)
    print(v.size(), v)
    output = v.sum()
    output.backward()
    print(t.grad)


    u_0 = torch.rand(N, C, T, H, W)
    print(u_0.requires_grad)
    conv = torch.nn.Conv3d(C, C, kernel_size=(3,3,3), padding=1)
    u_1 = conv(u_0)
    print(u_1.requires_grad)

