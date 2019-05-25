"""
Runtime Configuration File
"""

__all__ = [
    "__STRICT__",
    "__VERBOSE__", "__VERY_VERBOSE__"
    ]

# ----------------------------------------------------------------- #
#           Runtime Settings for Python Scripts                     #
# ----------------------------------------------------------------- #
# NOTE: __debug__ parameter cannot be overwritten in normal Python 
# scrips, so we don't modify it here. You should specify it via -O
# parameter when you run Python intepreter.
# * in __test__ mode, all scripts will perform self-test
# * in __profile__ mode, all scripts will counting execution time and 
#   report time break down
# * in __strict__ mode, all scripts will have more strict santity check 
#   to make sure you use it as the intended way
# * in __verbose__ mode, all scripts will use Python logging module to 
#   log some critical information. while you may specify whether the 
#   logging module needs to dump the log info or not.
# * in __vverbose__ (very verbose) mode, all scripts will print detailed
#   information (including but not limited to __verbose__ information)
#   in stdout.
__STRICT__ = True
__VERBOSE__ = True
__VERY_VERBOSE__ = True
__VERY_VERY_VERBOSE__ = False

def show_config():
    """
    Display package configuration
    """
    print("------------------------------------------------")
    print("Configuration For Package [vdataset]")
    print("__STRICT__", __STRICT__)
    print("__VERBOSE__", __VERBOSE__)
    print("__VERY_VERBOSE__", __VERY_VERBOSE__)
    print("------------------------------------------------")

if __name__ == "__main__":
    show_config()
