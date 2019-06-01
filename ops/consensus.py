import torch
import math


class AverageConsensusFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.mean(dim=1, keepdim=True)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        shape = input.size()
        grad_in = grad_output.expand(shape) / float(shape[1])
        return grad_in


average_consensus = AverageConsensusFunction.apply


class Consensus(torch.nn.Module):

    def __init__(self, type, dim=1):
        super(Consensus, self).__init__()
        self.type = type if type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        if self.type == 'avg':
            return average_consensus(input)
        else:
            raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + \
            "(type={}, dim={})".format(self.type, self.dim)
