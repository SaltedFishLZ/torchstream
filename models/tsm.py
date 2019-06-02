import torch
import torch.nn as nn

import torchvision

from ops.shift import TemporalShift
from ops.pool import TemporalPool

def make_temporal_shift(net, seg_num, fold_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        seg_num_list = [seg_num, seg_num // 2, seg_num // 2, seg_num // 2]
    else:
        seg_num_list = [seg_num] * 4
    assert seg_num_list[-1] > 0
    print('=> seg_num per stage: {}'.format(seg_num_list))

    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            def make_block_temporal(stage, seg_num):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    ## insert a temporal shift moduel before the block
                    blocks[i] = nn.Sequential(
                                        TemporalShift(seg_num=seg_num,
                                                      fold_div=fold_div),
                                        b
                                        )
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, seg_num_list[0])
            net.layer2 = make_block_temporal(net.layer2, seg_num_list[1])
            net.layer3 = make_block_temporal(net.layer3, seg_num_list[2])
            net.layer4 = make_block_temporal(net.layer4, seg_num_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                print('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, seg_num):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        blocks[i].conv1 = nn.Sequential(
                                        TemporalShift(seg_num=seg_num,
                                                      fold_div=fold_div),
                                        b.conv1
                                        )
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, seg_num_list[0])
            net.layer2 = make_block_temporal(net.layer2, seg_num_list[1])
            net.layer3 = make_block_temporal(net.layer3, seg_num_list[2])
            net.layer4 = make_block_temporal(net.layer4, seg_num_list[3])
        else:
            raise NotImplementedError(place)
    
    else:
        raise NotImplementedError("Backbone netowrk {}".format(net))


def make_temporal_pool(net, seg_num):
    if isinstance(net, torchvision.models.ResNet):
        print('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, seg_num)
    else:
        raise NotImplementedError

