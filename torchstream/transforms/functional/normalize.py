import torch

from .blob import _is_vtensor

def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor video with mean and standard deviation.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor video of size [C][T][H][W] to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    Returns:
        Tensor: Normalized vtensor.
    """
    if not _is_vtensor(tensor):
        raise TypeError('tensor is not a vtensor.')

    if not inplace:
        tensor = tensor.clone()


    if torch.__version__ >= "1.0.0": 
        mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    else:
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        
    tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return tensor
