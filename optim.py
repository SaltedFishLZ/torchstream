"""
"""
import torch

def get_optim_policies(model):

    first_conv_weight = []
    first_conv_bias = []
    
    normal_conv_weights = []
    normal_conv_biases = []
    
    normal_fc_weights = []
    normal_fc_biases = []

    normal_bns = []
    
    custom_ops = []
    
    conv_cnt = 0
    bn_cnt = 0

    for m in model.modules():

        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
            ps = list(m.parameters())
            conv_cnt += 1
            if conv_cnt == 1:
                first_conv_weight.append(ps[0])
                if len(ps) == 2:
                    first_conv_bias.append(ps[1])
            else:
                normal_conv_weights.append(ps[0])
                if len(ps) == 2:
                    normal_conv_biases.append(ps[1])

        elif isinstance(m, torch.nn.Linear):
            ps = list(m.parameters())
            normal_fc_weights.append(ps[0])
            if len(ps) == 2:
                normal_fc_biases.append(ps[1])

        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            bn_cnt += 1
            # later BN's are frozen
            if not model._enable_pbn or bn_cnt == 1:
                normal_bns.extend(list(m.parameters()))

        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}.".format(type(m)))
        else:
            raise NotImplementedError("Unknown Layer")

    return [
        {
            'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
            'name': "first_conv_weight"
        },
        {
            'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
            'name': "first_conv_bias"
        },
        {
            'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
            'name': "custom_ops"
        }
    ]