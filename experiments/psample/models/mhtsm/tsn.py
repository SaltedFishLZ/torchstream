import sys
import collections

from torch import nn
from torchstream.ops import Consensus

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
else:
    Sequence = collections.abc.Sequence


class TSN(nn.Module):
    """
    Args:
        input_size (tuple): (T, H, W), shape of the input blob.
        channel == 3 only
    """
    def __init__(self, cls_num, input_size, slices,
                 base_model="resnet50",
                 dropout=0.8, partial_bn=True,
                 **kwargs):
        print("Multi-Head TSN implementation")
        super(TSN, self).__init__()

        assert isinstance(input_size, (tuple, list)), TypeError
        assert len(input_size) == 3, ValueError

        self.cls_num = cls_num
        self.input_size = tuple(input_size)
        assert isinstance(slices, Sequence), TypeError
        for _s in slices:
            assert isinstance(_s, Sequence), TypeError
            for idx in _s:
                assert isinstance(idx, int), TypeError
                assert idx in range(self.seg_num), \
                    ValueError("{} out of range({})".format(
                        idx, self.seg_num
                        ))
        self.slices = slices
        self.branches = len(slices)

        self.dropout = dropout
        self._enable_pbn = partial_bn
        self._prepare_base_model(base_model)

        self.consensuses = nn.ModuleList([Consensus("avg"), ] * self.branches)

    @property
    def seg_num(self):
        return self.input_size[0]

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
        raise NotImplementedError

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters
        """
        super(TSN, self).train(mode)
        count = 0

        if self._enable_pbn:
            print("Freezing BatchNorm except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    count += 1
                    if count >= 2:
                        # shutdown update in frozen mode for BN layers
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def forward(self, input):

        # input shape checking
        shape = input.size()
        assert len(shape) == 5, ValueError
        N, C, T, H, W = shape
        assert (T, H, W) == self.input_size, ValueError

        # N C T H W -> N T C H W
        input = input.permute(0, 2, 1, 3, 4).contiguous()
        # merge time to batch
        input = input.view(N * T, C, H, W)

        base_outs = self.base_model(input)

        outputs = []
        for _i in range(self.branches):
            _s = self.slices[_i]
            T = len(_s)
            base_out = base_outs[_i]
            # reshape:
            base_out = base_out.view((-1, T) + base_out.size()[1:])
            output = self.consensuses[_i](base_out)
            outputs.append(output.squeeze(1))

        return outputs

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

            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "Unkown atomic module type: {}.".format(type(m))
                    )

        return [
            {"params": first_conv_weight, "lr_mult": 1, "decay_mult": 1,
             "name": "first_conv_weight"},
            {"params": first_conv_bias, "lr_mult": 2, "decay_mult": 0,
             "name": "first_conv_bias"},
            {"params": normal_weight, "lr_mult": 1, "decay_mult": 1,
             "name": "normal_weight"},
            {"params": normal_bias, "lr_mult": 2, "decay_mult": 0,
             "name": "normal_bias"},
            {"params": bn, "lr_mult": 1, "decay_mult": 0,
             "name": "BN scale/shift"},
            {"params": custom_ops, "lr_mult": 1, "decay_mult": 1,
             "name": "custom_ops"},
            # for fc
            {"params": lr5_weight, "lr_mult": 5, "decay_mult": 1,
             "name": "lr5_weight"},
            {"params": lr10_bias, "lr_mult": 10, "decay_mult": 0,
             "name": "lr10_bias"},
        ]
