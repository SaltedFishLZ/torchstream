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

def config2transform(cfg):
    """
    """
    package = "torchstream.transforms"
    if "package" in cfg:
        package = cfg["package"]

    transform_package = importlib.import_module(package)
    transform_class = getattr(transform_package, cfg["name"])

    transform = None
    if "argv" in cfg:
        argv = cfg["argv"]
        transform = transform_class(**argv)
    else:
        transform = transform_class()

    return transform


def config2framesampler(cfg):
    """
    """
    package = "torchstream.io.framesampler"
    if "package" in cfg:
        package = cfg["package"]

    framesampler_package = importlib.import_module(package)
    framesampler_class = getattr(framesampler_package, cfg["name"])

    framesampler = None
    if "argv" in cfg:
        argv = cfg["argv"]
        framesampler = framesampler_class(**argv)
    else:
        framesampler = framesampler_class()

    return framesampler


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

    dataset = None
    if "argv" in cfg:
        argv = cfg["argv"]
        dataset = dataset_class(**argv)
    else:
        dataset = dataset_class()

    if "holdout" in cfg:
        holdout = cfg["holdout"]
        path = holdout["path"]
        remove = holdout["remove"]
        holdout_index = torch.load(path)
        dataset.holdout(holdout_index, remove=remove)

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

    model = None
    if "argv" in cfg:
        argv = cfg["argv"]
        model = model_class(**argv)
    else:
        model = model_class()

    sync_bn = False
    if "sync_bn" in cfg:
        sync_bn = cfg["sync_bn"]
    if sync_bn:
        print("Using Sync BN...")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

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


def config2lrscheduler(optimizer, cfg):
    """
    """
    package = "torch.optim.lr_scheduler"
    if "package" in cfg:
        package = cfg["package"]

    lr_scheduler_package = importlib.import_module(package)
    lr_scheduler_class = getattr(lr_scheduler_package, cfg["name"])

    argv = {}
    if "argv" in cfg:
        argv = cfg["argv"]

    lr_scheduler = lr_scheduler_class(optimizer, **argv)
    return lr_scheduler
    
