from collections import OrderedDict

import torchvision
from torch import nn
from torch.nn.init import normal, constant

from torchstream.ops import Consensus, Identity
from torchstream.transforms import *


class TSN(nn.Module):
    """
    Args:
        input_size (tuple): (T, H, W), shape of the input blob. channel == 3 only
    """
    def __init__(self, cls_num, input_size,
                 base_model='resnet50', dropout=0.8, partial_bn=True,
                 use_softmax=False, **kwargs):

        super(TSN, self).__init__()

        assert isinstance(input_size, (tuple, list)), TypeError
        assert len(input_size) == 3, ValueError

        self.cls_num = cls_num

        self.input_size = tuple(input_size)

        self.dropout = dropout
        self.use_softmax = use_softmax

        self._prepare_base_model(base_model)

        self.consensus = Consensus("avg")

        if self.use_softmax:
            self.softmax = nn.Softmax()

        if partial_bn:
            self.partialBN(True)


    def partialBN(self, enable):
        self._enable_pbn = enable


    def __repr__(self, idents=0):
        format_string = self.__class__.__name__
        format_string += "\n\tbase model:    {}"
        format_string += "\n\tclass number:  {}"
        format_string += "\n\t(T, H, W):     {}"
        format_string += "\n\tdropout ratio: {}"
        return format_string.format(self.base_model, self.cls_num,
                                    self.input_size, self.dropout)


    def _prepare_base_model(self, base_model):
        """
        """
        if 'resnet' in base_model or 'vgg' in base_model:

            self.base_model = getattr(torchvision.models, base_model)(True)
            
            ## replace the classifier
            feature_dim = self.base_model.fc.in_features
            if self.dropout > 0:
                new_fc = nn.Sequential(OrderedDict([
                    ("dropout", nn.Dropout(p=self.dropout)),
                    ("fc", nn.Linear(feature_dim, self.cls_num))
                    ]))
            else:
                new_fc = nn.Linear(feature_dim, self.cls_num)
            self.base_model.fc = new_fc
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))



    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters
        """
        super(TSN, self).train(mode)
        count = 0

        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= 2:
                        # shutdown update in frozen mode for BN layers
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False


    def forward(self, input):

        ## input shape checking
        shape = input.size()
        assert len(shape) == 5, ValueError
        N, C, T, H, W = shape
        assert (T, H, W) == self.input_size, ValueError

        ## N C T H W -> N T C H W
        input = input.permute(0, 2, 1, 3, 4).contiguous()
        ## merge time to batch
        input = input.view(N * T, C, H, W)

        base_out = self.base_model(input)

        # if self.dropout > 0:
        #     base_out = self.new_fc(base_out)

        if self.use_softmax:
            base_out = self.softmax(base_out)

        ## reshape:
        base_out = base_out.view((-1, T) + base_out.size()[1:])
        
        output = self.consensus(base_out)
        return output.squeeze(1)



    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([MultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   RandomHorizontalFlip()])
        else:
            raise ValueError

