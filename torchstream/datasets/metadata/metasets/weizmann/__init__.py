"""
Weizmann Dataset
http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html
"""
__dataset__ = "Weizmann"    # dataset name
__layout__ = "UCF101"       # dataset file layout
__all__ = [
    "__dataset__", "__layout__",
    "__LABELS__", "__SAMPLES_PER_LABEL__",
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]
from . import path
__all__ += path.__all__
from .split import TrainsetFilter, ValsetFilter, TestsetFilter
from .path import *

# ------------------------------------------------------------------------ #
#             Labels, Corresponding CIDs & Sample Counts                   #
# ------------------------------------------------------------------------ #

__LABELS__ = {
    'bend'  : 1,
    'jack'  : 2,
    'jump'  : 3,
    'pjump' : 4,
    'run'   : 5,
    'side'  : 6,
    'skip'  : 7,
    'walk'  : 8,
    'wave1' : 9,    # wave with 1 hand
    'wave2' : 10    # wave with 2 hands
}

__SAMPLES_PER_LABEL__ = {
    "bend"  :   9,
    "jack"  :   9,
    "jump"  :   9,
    "pjump" :   9,
    "run"   :   10,
    "side"  :   9,
    "skip"  :   10,
    "walk"  :   10,
    "wave1" :   9,
    "wave2" :   9,
}
