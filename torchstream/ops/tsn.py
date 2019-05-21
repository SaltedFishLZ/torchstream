import copy

import torch


__supported_consensuses__ = ["avg", "identity"]

class SegmentConsensus(torch.autograd.Function):
    '''
    Segment Consensus Operator in paper [] Section 
    '''
    def __init__(self, consensus, dim=1):
        '''
        - consensus : consensus type
        - dim : which dimension the aggregation happens
        '''
        # santity check
        assert (consensus in __supported_consensuses__), \
            "Unsupported consensus type"
        # main stuff
        self.consensus = copy.deepcopy(consensus)
        self.dim = copy.deepcopy(dim)
        self.shape = None

    def forward(self, input_tensor):
        '''
        - input : PyTorch tensor
        '''
        self.shape = input_tensor.size()
        if self.consensus == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus == 'identity':
            output = input_tensor
        else:
            output = None
        return output

    def backward(self, grad_output):
        if self.consensus == 'avg':
            grad_in = grad_output.expand(self.shape) \
                / float(self.shape[self.dim])
        elif self.consensus == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus = consensus if consensus != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus, self.dim)(input)
