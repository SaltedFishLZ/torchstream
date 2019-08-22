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
    'bend'  : 0,
    'jack'  : 1,
    'jump'  : 2,
    'pjump' : 3,
    'run'   : 4,
    'side'  : 5,
    'skip'  : 6,
    'walk'  : 7,
    'wave1' : 8,    # wave with 1 hand
    'wave2' : 9     # wave with 2 hands
}

__SAMPLES_PER_LABEL__ = {
    "bend"  :   [9  , 9 ] ,
    "jack"  :   [9  , 9 ] ,
    "jump"  :   [9  , 9 ] ,
    "pjump" :   [9  , 9 ] ,
    "run"   :   [10 , 10] ,
    "side"  :   [9  , 9 ] ,
    "skip"  :   [10 , 10] ,
    "walk"  :   [10 , 10] ,
    "wave1" :   [9  , 9 ] ,
    "wave2" :   [9  , 9 ] ,
}
