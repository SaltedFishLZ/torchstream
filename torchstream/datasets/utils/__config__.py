"""
Runtime Configuration File For Package [vdataset.utils]
"""

__all__ = [
    "__STRICT__",
    "__VERBOSE__", "__VERY_VERBOSE__", "__VERY_VERY_VERBOSE__",
    "__TQDM__"
    ]

from ..__config__ import *

# overide global configuration
__STRICT__ = True
__VERBOSE__ = False
__VERY_VERBOSE__ = False
__VERY_VERY_VERBOSE__ = False
__TQDM__ = True             # enable tqdm

def show_config():
    """
    Display package configuration
    """
    print("------------------------------------------------")
    print("Configuration For Package [vdataset.utils]")
    print("__STRICT__", __STRICT__)
    print("__VERBOSE__", __VERBOSE__)
    print("__VERY_VERBOSE__", __VERY_VERBOSE__)
    print("__VERY_VERY_VERBOSE__", __VERY_VERY_VERBOSE__)
    print("__TQDM__", __TQDM__)
    print("------------------------------------------------")

if __name__ == "__main__":
    show_config()
