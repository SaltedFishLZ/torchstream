"""Video Dataset Module
"""
__authors__ = ["Zheng Liang @ Peking University", "Bernie Wang @ UC Berkeley"]
__author__ = __authors__
__version__ = "0.1"
__all__ = ["HMDB51", "UCF101", "JesterV1", "SomethingSomethingV1", "Kinetics400"]

from .hmdb51 import HMDB51
from .ucf101 import UCF101
from .kinetics400 import Kinetics400
from .jesterv1 import JesterV1
from .sthsthv1 import SomethingSomethingV1
# from .sthsthv2 import SomethingSomethingV2
# from .tinysthsthv1 import TinySomethingSomethingV1
# from .tinysthsthv2 import TinySomethingSomethingV2
