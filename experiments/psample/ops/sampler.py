"""
"""
import torch
 
class UniformIntervalSampler(torch.nn.Module):
    """
    """
    def __init__(self, size, dim, mode="edge"):
        super(UniformIntervalSampler, self).__init__()
        assert mode in ["center", "edge"], NotImplementedError
        self.size = size
        self.dim = dim
        self.mode = mode

    def forward(self, input):
        selections = None
        length = input.size(self.dim) - 1

        if self.mode == "edge":
            interval = length / (self.size - 1)
            selections = (torch.arange(self.size).float() * interval).long()
        elif self.mode == "center":
            interval = length / self.size
            selections = (torch.arange(self.size).float() + 0.5).long()
        else:
            raise NotImplementedError
        return torch.index_select(input, self.dim, selections)

if __name__ == "__main__":
    N, C, T, H, W = 1, 3, 33, 224, 224
    volume = torch.randn(N, C, T, H, W)
 
    edge_sampler = UniformIntervalSampler(size=5, dim=2, mode="edge")
    output = edge_sampler(volume)
    print((output == volume[:, :, [0, 8, 16, 24, 32],:, :]).prod())