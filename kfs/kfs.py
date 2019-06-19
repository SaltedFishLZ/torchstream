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

    def forward(self, x):
        
        out = self.pool(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.pool(out)
    
        out = self.avgpool(out)

        
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)

        return out


class Wrapper(nn.Module):
    def __init__(self):
        super(Wrapper, self).__init__()

        self.classifier = TSM(cls_num=174, input_size=[8, 224, 224])
        self.selector = KFS(output_size=8)
        self.interpolate = TemporalInterpolationModule()

    def freeze_classifier(self):
        for p in self.classifier.parameters():
            p.requires_grad_(False)


    def forward(self, x):

        index = self.selector(x)

        out = x.permute(0, 2, 1, 3, 4).contiguous()
        out = self.interpolate(out, index)
        out = out.permute(0, 2, 1, 3, 4).contiguous()

        out = self.classifier(out)

        return out


if __name__ == "__main__":
    kfs = KFS(output_size=8)
    input = torch.rand(4, 3, 9, 224, 224)
    index = kfs(input)
    print(index)

    wrapper = Wrapper()
    print(list(wrapper.modules()))
    print(len(list(wrapper.classifier.parameters())))
    for p in wrapper.classifier.base_model.bn1.parameters():
        print(p.size())

    x = wrapper(input)
