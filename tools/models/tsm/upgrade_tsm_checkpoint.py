"""Upgrade TSM model
ResNet:
    conv2 -> conv2.conv
    conv3 -> conv3.conv
"""
import sys
import copy
import torch


def process_resnets(src):
    """
    Args:
        src: source model state dict
    Return:
        processed model state dict
    """
    dst = copy.deepcopy(src)
    old_keys = list(dst.keys())
    for key in old_keys:
        val = dst[key]
        new_key = key
        if "layer" in key:
            if ".conv2" in key:
                new_key = new_key.replace('.conv2', '.conv2.conv')
            if ".conv3" in key:
                new_key = new_key.replace('.conv3', '.conv3.conv')
            if "downsample.0" in key:
                new_key = new_key.replace('downsample.0', 'downsample.0.conv')
        del dst[key]
        dst[new_key] = val
    
    return dst


if __name__ == "__main__":
    if len(sys.argv) == 3:
        src = sys.argv[1]
        dst = sys.argv[2]

        ckpt = torch.load(src, map_location="cpu")

        src_model_state_dict = ckpt["model_state_dict"]

        dst_model_state_dict = process_resnets(src_model_state_dict)

        ckpt["model_state_dict"] = dst_model_state_dict

        torch.save(ckpt, dst)
    else:
        print("cmd src dst")
