"""Blob Transformation
Default np.ndarray layout [T][H][W][C]
Default Tensor data layout [C][T][H][W]
"""
__all__ = ["to_tensor", "to_varray"]

import torch
import numpy as np

def _is_vtensor(x):
    return(
        torch.is_tensor(x)
        and (x.ndimension() == 4)
    )

def _is_varray(x):
    return(
        isinstance(x, np.ndarray)
        and (x.ndim == 4)
    )

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
    varray = np.transpose(varray, (3, 0, 1, 2))
    vtensor = torch.from_numpy(varray)

    # backward compatibility
    if isinstance(vtensor, torch.ByteTensor):
        return vtensor.float().div(255)
    else:
        return vtensor

def to_varray(vid):
    """Convert tensor to ``varray``
    """
    if not _is_vtensor(vid):
        raise TypeError
    
    if isinstance(vid, torch.FloatTensor):
        vid = vid.mul(255).byte()

    # CTHW -> THWC
    varray = vid.numpy()
    varray = np.transpose(varray, (1, 2, 3, 0))

    return varray