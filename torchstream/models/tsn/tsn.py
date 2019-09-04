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
                 base_model="resnet50", dropout=0.8, partial_bn=True,
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

        self._enable_pbn = partial_bn

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
            new_fc = nn.Sequential(OrderedDict([
                ("dropout", nn.Dropout(p=self.dropout)),
                ("fc", nn.Linear(feature_dim, self.cls_num))
                ]))

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
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
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
            return torchvision.transforms.Compose([
                MultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                RandomHorizontalFlip()
                ])
        else:
            raise ValueError

    def get_optim_policies(self, fc_lr5=False):
        
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])

            elif isinstance(m, nn.Linear):
                ps = list(m.parameters())
                if fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, nn.modules.batchnorm._BatchNorm):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            
            # elif isinstance(m, nn.BatchNorm3d):
            #     bn_cnt += 1
            #     # later BN's are frozen
            #     if not self._enable_pbn or bn_cnt == 1:
            #         bn.extend(list(m.parameters()

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]
