"""
"""
import torch
import torch.nn.functional as F


class TemporalInterpolationFunction(torch.autograd.Function):
    """
    Args:

    """
    @staticmethod
    def forward(ctx, input, index):
        """
        input [N][T][C][H][W]
        """
        T = input.size(dim)
        left = torch.floor(index * T).long()
        right = left + 1
        last = (left >= (T - 1))
        f_left = torch.index_select(input=input, )
        output = last.float() * 


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