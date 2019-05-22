"""
Runtime Configuration File For Package [vdataset.metadata.metasets]
"""

from ..__config__ import *

__all__ = [
    "__STRICT__",
    "__VERBOSE__", "__VERY_VERBOSE__", "__VERY_VERY_VERBOSE__",
    "__TQDM__"
    ]

# overide global configuration
__STRICT__ = True
__VERBOSE__ = True
__VERY_VERBOSE__ = True
__VERY_VERY_VERBOSE__ = True
__TQDM__ = True             # enable tqdm

def show_config():
    """
    Display package configuration
    """
    print("------------------------------------------------")
    print("Configuration For Package [vdataset.metadata.metasets]")
    print("__STRICT__", __STRICT__)
    print("__VERBOSE__", __VERBOSE__)
    print("__VERY_VERBOSE__", __VERY_VERBOSE__)
    print("__VERY_VERY_VERBOSE__", __VERY_VERY_VERBOSE__)
    print("__TQDM__", __TQDM__)
    print("------------------------------------------------")

if __name__ == "__main__":
    show_config()
