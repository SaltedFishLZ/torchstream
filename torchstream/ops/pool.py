
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalPool(nn.Module):
    def __init__(self, net, seg_num):
        super(TemporalPool, self).__init__()
        self.net = net
        self.seg_num = seg_num

    def forward(self, x):
        x = self.temporal_pool(x, seg_num=self.seg_num)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, seg_num):
        nt, c, h, w = x.size()
        n_batch = nt // seg_num
        x = x.view(n_batch, seg_num, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x



