from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn

from .tsn import TSN
from torchstream.ops import Consensus, Identity, TemporalShift, TemporalPool
from torchstream.transforms import *


def make_temporal_shift(net, seg_num, fold_div=8,
                        place='blockres',
                        temporal_pool=False,
                        verbose=False
                       ):
    """
    """
    if temporal_pool:
        seg_num_list = [seg_num, seg_num // 2, seg_num // 2, seg_num // 2]
    else:
        seg_num_list = [seg_num] * 4
    assert seg_num_list[-1] > 0
    if verbose:
        print('=> seg_num per stage: {}'.format(seg_num_list))

    if isinstance(net, torchvision.models.ResNet):
        ## insert a temporal shift moduel before the block
        if place == 'block':

            def make_block_temporal(stage, seg_num):
                old_blocks = list(stage.children())
                new_blocks = []
                if verbose:
                    print('=> Processing stage with {} blocks'.format(len(old_blocks)))
                for block in old_blocks:
                    shift = TemporalShift(seg_num=seg_num, fold_div=fold_div)
                    new_blocks.append([shift, block])
                return nn.Sequential(*(new_blocks))

            net.layer1 = make_block_temporal(net.layer1, seg_num_list[0])
            net.layer2 = make_block_temporal(net.layer2, seg_num_list[1])
            net.layer3 = make_block_temporal(net.layer3, seg_num_list[2])
            net.layer4 = make_block_temporal(net.layer4, seg_num_list[3])
        ##
        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                if verbose:
                    print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, seg_num):
                blocks = list(stage.children())
                if verbose:
                    print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        shift = TemporalShift(seg_num=seg_num, fold_div=fold_div)
                        blocks[i].conv1 = nn.Sequential(OrderedDict([
                                        ("shift", shift),
                                        ("conv", b.conv1)
                                        ]))
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, seg_num_list[0])
            net.layer2 = make_block_temporal(net.layer2, seg_num_list[1])
            net.layer3 = make_block_temporal(net.layer3, seg_num_list[2])
            net.layer4 = make_block_temporal(net.layer4, seg_num_list[3])
        ## unknown place
        else:
            raise NotImplementedError(place)

    else:
        raise NotImplementedError("Backbone netowrk {}".format(net))


def make_temporal_pool(net, seg_num, verbose=False):
    if isinstance(net, torchvision.models.ResNet):
        if verbose:
            print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, seg_num)
    else:
        raise NotImplementedError





class TSM(TSN):

    def __init__(self, cls_num, input_size,
                 base_model='resnet50', dropout=0.8, partial_bn=True,
                 use_softmax=False,
                 fold_div=8, shift_place='blockres',
                 temporal_pool=False, non_local=False,
                 **kwargs
                ):

        super(TSM, self).__init__(
            cls_num=cls_num, input_size=input_size,
            base_model=base_model, dropout=dropout, partial_bn=partial_bn,
            use_softmax=use_softmax,
            **kwargs
            )

        self.fold_div = fold_div
        self.shift_place = shift_place
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        ## insert temporal shift modules
        seg_num, _, _ = input_size
        make_temporal_shift(net=self.base_model,
                            seg_num=seg_num, fold_div=self.fold_div,
                            place=self.shift_place,
                            temporal_pool=self.temporal_pool)


    def __repr__(self, idents=0):
        format_string = self.__class__.__name__
        format_string += "\n\tbase model:    {}"
        format_string += "\n\tclass number:  {}"
        format_string += "\n\t(T, H, W):     {}"
        format_string += "\n\tdropout ratio: {}"
        format_string += "\n\tshift_place:   {}"
        format_string += "\n\tfold_div:      {}"
        return format_string.format(self.base_model, self.cls_num,
                                    self.input_size, self.dropout,
                                    self.shift_place, self.fold_div)



if __name__ == "__main__":
    net = TSM(cls_num=101, input_size=(8,224,224))
    print(net.state_dict().keys())
