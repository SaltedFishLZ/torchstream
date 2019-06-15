import sys

import torch

import utils

def transform_tsm_weight(checkpoint):

    model_state_dict = checkpoint["state_dict"]

    old_keys = list(model_state_dict.keys())
    for key in old_keys:
        val = model_state_dict[key]
        new_key = key.replace("module.", "")
        if "conv1.net" in new_key:
            new_key = new_key.replace('conv1.net', 'conv1.conv')
        if "new_fc" in new_key:
            new_key = new_key.replace('new_fc', 'base_model.fc.fc')
        del model_state_dict[key]
        model_state_dict[new_key] = val

    return checkpoint

if __name__ == "__main__":
    if len(sys.argv) == 2:
        checkpoint = torch.load(sys.argv[1])
        checkpoint = transform_tsm_weight(checkpoint)
        torch.save(checkpoint, sys.argv[1])
    else:
        print("cmd <pth path>")
