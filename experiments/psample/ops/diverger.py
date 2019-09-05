import torch

class NetworkDiverger(torch.nn.Module):
    """
    Args:
        branches (nn.ModuleList): a ModuleList for all branches.
    """
    def __init__(self, branches):
        super(NetworkDiverger, self).__init__()
        self.branches = branches


    def forward(self, input):
        output = []
        for branch in self.branches:
            output.append(branch)
        return output