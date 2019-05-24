"""Tensor Transformation
Default np.ndarray layout [T][H][W][C]
Default Tensor data layout [C][T][H][W]
"""
__all__ = ["to_tensor"]

import torch
import numpy as np

from .typing import _is_varray

def to_tensor(varray):
    """Convert a ``varray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        varray (ndarray): video array to be converted to tensor.
    Returns:
        Tensor: Converted video.
    """
    if not _is_varray(varray):
        raise TypeError('varray should be ndarray. Got {}'.format(varray))

    # handle numpy array
    vtensor = torch.from_numpy(varray.transpose((3, 0, 1, 2)))
    # backward compatibility
    if isinstance(vtensor, torch.ByteTensor):
        return vtensor.float().div(255)
    else:
        return vtensor