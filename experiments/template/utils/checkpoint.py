import os, time, sys, math, copy
import subprocess, shutil

import torch

def to_cpu(state_dict, inplace=False):
    """Transfer a state_dict back to CPU
    """
    if not isinstance(state_dict, dict):
        return state_dict

    ret = state_dict
    if not inplace:
        ret = copy.deepcopy(state_dict)

    for key in ret:
        if isinstance(ret[key], torch.Tensor):
            ret[key] = ret[key].cpu()
        elif isinstance(ret[key], dict):
            ret[key] = to_cpu(ret[key], inplace=True)
        elif isinstance(ret[key], (list, tuple)):
            for element in ret[key]:
                element = to_cpu(element, inplace=True)

    return ret


def remove_prefix_in_keys(state_dict, prefix="module."):
    """Remove certain prefix string in the keys of given state_dict
    This is designed for DataParallel wrappers
    """
    old_keys = list(state_dict.keys())
    for key in old_keys:
        val = state_dict[key]
        new_key = key.replace(prefix, "")
        del state_dict[key]
        state_dict[new_key] = val


def save_checkpoint(checkpoint,
                    dir_path="checkpoints",
                    pth_name="model",
                    is_best=False,
                    **kwargs):
    """
    """
    os.makedirs(dir_path, exist_ok=True)
    ckpt_path = os.path.join(dir_path, pth_name)
    ckpt_path = os.path.realpath(ckpt_path)
    ckpt_path = os.path.expandvars(ckpt_path)
    ckpt_path = os.path.expanduser(ckpt_path)
    torch.save(checkpoint, ckpt_path + ".pth")
    if is_best:
        shutil.copyfile(ckpt_path + ".pth", ckpt_path + ".best.pth")


def load_checkpoint(dir_path="checkpoints", pth_name="model",
                    **kwargs):
    """
    """
    ckpt_path = os.path.join(dir_path, pth_name + ".pth")
    ckpt_path = os.path.realpath(ckpt_path)
    ckpt_path = os.path.expandvars(ckpt_path)
    ckpt_path = os.path.expanduser(ckpt_path)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path,
                                map_location="cpu")
        return checkpoint
    else:
        return None
