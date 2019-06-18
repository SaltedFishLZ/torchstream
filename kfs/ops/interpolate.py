"""
"""
import torch
import torch.nn.functional as F


class TemporalInterpolationFunction(torch.autograd.Function):
    """
    Args:

    """
    @staticmethod
    def forward(ctx, u, i):
        """
        u [N][Ti][C][H][W]
        i [N][To], value belongs to [0, 1]
        """
        N, Ti, C, H, W = u.size()
        N, To = i.size()
        L = Ti - 1

        i_l = torch.floor(i * L).long()
        i_r = i_l + 1
        
        u_l = torch.empty(N, To, C, H, W)
        u_r = torch.empty(N, To, C, H, W)
        for n in range(N):
            u_l[n, :, :, :, :] = u[n, i_l[n], :, :, :]
            u_r[n, :, :, :, :] = u[n, i_r[n], :, :, :]

        v = torch.empty(N, To, C, H, W)
        for n in range(N):
            for t in range(To):
                v[n, t, :, :, :] = \
                    (i_r[n, t] - i[n, t] * L) * u_l[n, t] + \
                    (i[n, t] * L - i_l[n, t]) * u_r[n, t]
        
        return v


    @staticmethod
    def backward(ctx, grad_output):
        pass

        if ctx.needs_input_grad[0]:
            alpha = 
            left_mask = scatter
            left_mask.sum
            right_mask = scatter
            right_mask.sum
            grad_input = 
        
        if ctx.needs_input_grad[1]:
            grad_output * (input(left) - input(right)) / (r - l)

temporal_interpolation = TemporalInterpolationFunction.apply

if __name__ == "__main__":
    N, T, C, H, W = 1, 10, 3, 224, 224
    u = torch.empty(N, T, C, H, W)
    index = torch.Tensor([0.5])
    v = temporal_interpolation(u, index)
    output = v.sum()
    