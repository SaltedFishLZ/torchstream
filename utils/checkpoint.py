import os, time, sys, math
import subprocess, shutil

import torch

def save_checkpoint(state, is_best, dir_path, pth_name="model"):
    file_path = os.path.join(dir_path, pth_name)
    torch.save(state, file_path + ".pth")
    if is_best:
        shutil.copyfile(name, file_path + ".best.pth")
