import os, time, sys, math
import subprocess, shutil

import torch

def save_checkpoint(checkpoint, dir_path="checkpoints", pth_name="model",
                    is_best=False, **kwargs):
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
    ckpt_path = os.path.join(dir_path, pth_name + ".pth")
    ckpt_path = os.path.realpath(ckpt_path)
    ckpt_path = os.path.expandvars(ckpt_path)
    ckpt_path = os.path.expanduser(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    return checkpoint