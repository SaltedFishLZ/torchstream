"""Runtime Configuration
"""

__all__ = [
    "STRICT",
    "LOGGER_LEVEL"
    ]

# import global configuration
# from ..__config__ import *
import logging

# overide global configuration
STRICT = True
LOGGER_LEVEL = logging.ERROR


def show_config():
    """Display package configuration
    """
    print("-" * 128)
    print("package config [{}]".format(__file__))
    print("-" * 128)
    print("STRICT: ", STRICT)
    print("LOGGER_LEVEL: ", LOGGER_LEVEL)
    print("-" * 128)


if __name__ == "__main__":
    show_config()
