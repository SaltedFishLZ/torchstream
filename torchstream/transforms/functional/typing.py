import torch
import numpy as np


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
