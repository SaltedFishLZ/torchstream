import torch
import torch.nn as nn

from torchstream.models import TSM
from ops import TemporalInterpolationModule


class KFS(nn.Module):
    """
    """
    def __init__(self, input_size=(16, 224, 224), output_size=8, debug=False):
        super(KFS, self).__init__()
        self.debug = debug

        self.input_size = tuple(input_size)
        self.output_size = output_size

        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64,
                               kernel_size=(5, 5, 5), padding=(2, 0, 0),
                               stride=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(num_features=64)

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5, 5), padding=(2, 0, 0))
        self.bn2 = nn.BatchNorm3d(num_features=128)

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5, 5), padding=2)
        self.bn3 = nn.BatchNorm3d(num_features=256)

        self.conv4 = nn.Conv3d(in_channels=256, out_channels=128,
                               kernel_size=(5, 1, 1), padding=(2, 0, 0))
        self.bn4 = nn.BatchNorm3d(num_features=128)

        self.adptpool = nn.AdaptiveMaxPool3d(output_size=(input_size[0], 12, 12))

        self.fc1 = nn.Linear(in_features=128 * input_size[0] * 12 * 12,
                             out_features=1024)

        self.fc2 = nn.Linear(in_features=1024,
                             out_features=(self.output_size + 1))
        # self.fc2.weight.data.fill_(0.0)
        # self.fc2.bias.data.fill_(0.0)


    def forward(self, x):
        N, C, T, H, W = x.size()
        assert self.input_size == (T, H, W),\
            ValueError("Input size error: {} expected, {} got".
            format(self.input_size, (T, H, W)))

        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # if self.debug:
        #     print("KFS ReLU 2 Output", out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.adptpool(out)
        # if self.debug:
        #     print("KFS avgpool Output", out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)

        if self.debug:
            print("KFS Final Output", out)

        return out


class Wrapper(nn.Module):
    def __init__(self, cls_num, input_size, debug=False):
        super(Wrapper, self).__init__()
        self.debug = debug
        self.cls_num = cls_num
        self.input_size = input_size

        self.selector = KFS(input_size=input_size, output_size=8, debug=debug)
        self.interpolate = TemporalInterpolationModule(norm=True, mode="interval", debug=debug)
        self.classifier = TSM(cls_num=cls_num, input_size=[8, 224, 224], dropout=0.0, partial_bn=False)

    def freeze_classifier(self):
        self.classifier.eval()
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

    wrapper = Wrapper(cls_num=8, input_size=[9, 224, 224])
    print(list(wrapper.modules()))
    print(len(list(wrapper.classifier.parameters())))
    for p in wrapper.classifier.base_model.bn1.parameters():
        print(p.size())

    x = wrapper(input)
