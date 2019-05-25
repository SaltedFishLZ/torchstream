"""Blob Transformation
Default np.ndarray layout [T][H][W][C]
Default Tensor data layout [C][T][H][W]
"""
__all__ = ["to_tensor", "to_varray"]

import torch
import numpy as np

SHAPE_THWC = 0
SHAPE_TCHW = 1
SHAPE_CTHW = 2

def _is_vtensor(x):
    return(
        torch.is_tensor(x)
        and (x.ndimension() == 3)
    )

def _is_varray(x):
    return(
        isinstance(x, np.ndarray)
        and (x.ndim == 4)
    )

def to_tensor(varray, shape_out=SHAPE_CTHW):
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
    if shape_out == SHAPE_CTHW:
        varray = np.transpose(varray, (3, 0, 1, 2))
    elif shape_out == SHAPE_TCHW:
        varray = np.transpose(varray, (0, 3, 1, 2))
    else:
        raise NotImplementedError

    vtensor = torch.from_numpy(varray)
    # backward compatibility
    if isinstance(vtensor, torch.ByteTensor):
        return vtensor.float().div(255)
    else:
        return vtensor

def to_varray(vid, shape_out=SHAPE_THWC):
    """Convert tensor to ``varray``
    """
    if not _is_vtensor(vid):
        raise TypeError
    
    if isinstance(vid, torch.FloatTensor):
        vid = vid.mul(255).byte()

    if isinstance(vid, torch.Tensor):
        varray = vid.numpy()
        ## CTHW -> THWC
        if shape_out == SHAPE_TCHW:
            varray = np.transpose(varray, (1, 2, 3, 0))
        else:
            raise NotImplementedError
    
    return varray