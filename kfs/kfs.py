import torch
import torch.nn as nn

from torchstream.models import TSM
from ops import TemporalInterpolationModule

class KFS(nn.Module):
    """
    """
    def __init__(self, output_size):
        super(KFS, self).__init__()

        self.output_size = output_size

        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(5, 5, 5), padding=2)
        self.bn1 = nn.BatchNorm3d(num_features=64)

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(5, 5, 5), padding=2)
        self.bn2 = nn.BatchNorm3d(num_features=128)

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(5, 5, 5), padding=2)
        self.bn3 = nn.BatchNorm3d(num_features=256)

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(32, 2, 2))

        self.fc1 = nn.Linear(in_features=256 * 32 * 2 * 2, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=(self.output_size + 1))
        self.fc2.weight.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)

        # self.interpolate = TemporalInterpolationModule()

    def __repr__(self):
        string = self.__class__.__name__ + "\n"
        string += "\n"
        return string

    def forward(self, x):
        
        x = self.pool(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
    
        x = self.avgpool(x)

        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())

        print("before FC1")

        x = self.fc1(x)
        x = self.relu(x)

        print("before FC2")

        x = self.fc2(x)

        return x


class Wrapper(nn.Module):
    def __init__(self):
        super(Wrapper, self).__init__()

        self.basenet = TSM(cls_num = 127, input_size=[8, 224, 224])
        self.kfsnet = KFS(output_size = 8)
        self.interpolate = TemporalInterpolationModule()

    def forward(self, x):
        index = self.kfsnet(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.interpolate(x, index)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.basenet(x)


if __name__ == "__main__":
    kfs = KFS(output_size=8)
    input = torch.rand(4, 3, 9, 224, 224)
    index = kfs(input)
    print(index)

    wrapper = Wrapper()
    x = wrapper(input)
