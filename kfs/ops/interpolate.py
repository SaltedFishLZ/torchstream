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

        # if not (tr < Ti).prod().item():
        #     print(t)
        #     print(tr)
        #     raise ValueError

        alpha = t * (Ti - 1) - tl.float()

        # ul: left endpoint
        # ur: right endpoint
        ul = torch.zeros(N, To, C, H, W, device=u.device, requires_grad=False)
        ur = torch.zeros(N, To, C, H, W, device=u.device, requires_grad=False)
        for n in range(N):
            ul[n, :, :, :, :] = u[n, tl[n], :, :, :]
            if tr[n] < Ti:
                ur[n, :, :, :, :] = u[n, tr[n], :, :, :]
            else:
                ur[n, :, :, :, :] = u[n, tr[Ti - 1], :, :, :]

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
    N, T, C, H, W = 2, 10, 3, 224, 224

    # u = torch.ones(N, T, C, H, W)
    u = torch.empty(N, T, C, H, W)
    for t in range(T):
        # u[0, t] = t
        u[0, t] = t ** 2
        u[1, t] = 2 * t ** 2

    t = torch.Tensor([[0.5, 0], [0.5, 0.9999999]])
    t.requires_grad_(True)
    v = temporal_interpolation(u, t)
    print(v.size(), v)
    output = v.sum()
    output.backward()
    print(t.grad)

    u_0 = torch.rand(N, C, T, H, W)
    print(u_0.requires_grad)
    conv = torch.nn.Conv3d(C, C, kernel_size=(3, 3, 3), padding=1)
    u_1 = conv(u_0)
    print(u_1.requires_grad)
