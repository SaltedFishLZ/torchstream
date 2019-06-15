"""
Configuration Parser

Input Configuration Format:
If you need to import some class from certain module:
```json
    {
        "package": path to package
        "name": class name
        "argv": input arguments
    }
```
Otherwise:
```json
    {
        specify all arguments here
    }
```
"""
import importlib

import torch

import torchstream

def config2dataset(cfg):
    """Configuration -> Dataset
    Args:
        cfg (dict): dataset configuration
    """
    # default package
    package = "torchstream.datasets"
    if "package" in cfg:
        package = cfg["package"]

    dataset_package = importlib.import_module(package)
    dataset_class = getattr(dataset_package, cfg["name"])

    transform = None

    ## currently, we directly parse a linear list of transforms
    assert "transforms" in cfg["argv"], NotImplementedError
    transforms = []
    for _t_dict in cfg["argv"]["transforms"]:
        package = "torchstream.transforms"
        if "package" in _t_dict:
            package = _t_dict["package"]
        transform_package = importlib.import_module(package)
        transform_class = getattr(transform_package, _t_dict["name"])
        if "argv" in _t_dict:
            argv = _t_dict["argv"]
            _t = transform_class(**argv)
            transforms.append(_t)
        else:
            _t = transform_class()
            transforms.append(_t)
    transform = torchstream.transforms.Compose(transforms)
    # print(transform)
    del cfg["argv"]["transforms"]

    dataset = dataset_class(**cfg["argv"], transform=transform)

    return dataset


def config2dataloader(cfg):
    """Configuration -> DataLoader
    Args:
        dataset (PyTorch Dataset)
        cfg (dict): dataloader configuration
    """
    dataloader = torch.utils.data.DataLoader(**cfg)
    return dataloader


def config2model(cfg):
    """Configuration -> Neural Network Model
    Args:
        cfg (dict)
    """
    model_package = importlib.import_module(cfg["package"])
    model_class = getattr(model_package, cfg["name"])
    model = model_class(**cfg["argv"])
    return model


def config2criterion(cfg):
    """Configuration -> Loss / Criterion
    Args:
        cfg (dict)
    """
    package = "torch.nn"
    if "package" in cfg:
        package = cfg["package"]

    criterion_package = importlib.import_module(package)
    criterion_class = getattr(criterion_package, cfg["name"])
    
    argv = {}
    if "argv" in cfg:
        argv = cfg[argv]
    
    criterion = criterion_class(**argv)
    return criterion


def config2optimizer(cfg):
    """Configuration -> Training Optimizer
    Args:
        cfg (dict)
    """
    package = "torch.optim"
    if "package" in cfg:
        package = cfg["package"]

    optimizer_package = importlib.import_module(package)
    optimizer_class = getattr(optimizer_package, cfg["name"])

    argv = {}
    if "argv" in cfg:
        argv = cfg["argv"]

    optimizer = optimizer_class(**argv)
    return optimizer
