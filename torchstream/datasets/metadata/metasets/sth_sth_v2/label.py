"""Annotation data
"""
__all__ = [
    "__LABELS__", "__SAMPLES_PER_LABEL__",
    "__ANNOTATIONS__"
]

import os
import json
import pickle
import logging

from . import __config__
from .jsonparse import TRAINSET_JLIST, VALSET_JLIST, TESTSET_JLIST
from ...__const__ import UNKOWN_LABEL, UNKOWN_CID
from ....utils.filesys import touch_date

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# ---------------------------------------------------------------- #
#                  Configuring Python Logger                       #
# ---------------------------------------------------------------- #

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
if __config__.__VERY_VERY_VERBOSE__:
    logger.setLevel(logging.INFO)
elif __config__.__VERY_VERBOSE__:
    logger.setLevel(logging.WARNING)
elif __config__.__VERBOSE__:
    logger.setLevel(logging.ERROR)
else:
    logger.setLevel(logging.CRITICAL)


# ------------------------------------------------------------------------ #
#                   Labels and Corresponding CIDs                          #
# ------------------------------------------------------------------------ #

__LABELS__ = dict()
LIST = os.path.join(DIR_PATH, "something-something-v2-labels.json")
fin = open(LIST, "r")
__LABELS__ = dict(json.load(fin))
for _label in __LABELS__:
    __LABELS__[_label] = int(__LABELS__[_label])
fin.close()
CIDS = list(__LABELS__.values())

# ------------------------------------------------------------------------ #
#           Sample Number Per Class (Useful for Integrity Check)           #
# ------------------------------------------------------------------------ #
# 
__SAMPLES_PER_LABEL__ = {
    "Putting something on a surface"                                                                        : [50, 4081],                                                                      
    "Moving something up"                                                                                   : [50, 3750],                                                            
    "Covering something with something"                                                                     : [50, 3530],                                                                            
    "Pushing something from left to right"                                                                  : [50, 3442],                                                                            
    "Moving something down"                                                                                 : [50, 3242],                                                                
    "Pushing something from right to left"                                                                  : [50, 3195],                                                                            
    "Uncovering something"                                                                                  : [50, 3004],                                                            
    "Taking one of many similar things on the table"                                                        : [50, 2969],                                                                                        
    "Turning something upside down"                                                                         : [50, 2943],                                                                        
    "Tearing something into two pieces"                                                                     : [50, 2849],                                                                            
    "Putting something into something"                                                                      : [50, 2783],                                                                        
    "Squeezing something"                                                                                   : [50, 2631],                                                            
    "Throwing something"                                                                                    : [50, 2626],                                                            
    "Putting something next to something"                                                                   : [50, 2431],                                                                            
    "Poking something so lightly that it doesn't or almost doesn't move"                                    : [50, 2430],                                                                                                            
    "Pushing something so that it slightly moves"                                                           : [50, 2418],                                                                                    
    "Putting something similar to other things that are already on the table"                               : [50, 2339],                                                                                                                
    "Showing something behind something"                                                                    : [50, 2315],                                                                            
    "Moving something and something closer to each other"                                                   : [50, 2298],                                                                                            
    "Taking something out of something"                                                                     : [50, 2259],                                                                            
    "Plugging something into something"                                                                     : [50, 2252],                                                                            
    "Pushing something so that it falls off the table"                                                      : [50, 2240],                                                                                        
    "Hitting something with something"                                                                      : [50, 2234],                                                                        
    "Showing that something is empty"                                                                       : [50, 2209],                                                                        
    "Holding something in front of something"                                                               : [50, 2203],                                                                                
    "Something falling like a rock"                                                                         : [50, 2079],                                                                        
    "Moving something and something away from each other"                                                   : [50, 2062],                                                                                            
    "Tearing something just a little bit"                                                                   : [50, 2025],                                                                            
    "Lifting something with something on it"                                                                : [50, 2016],                                                                                
    "Stuffing something into something"                                                                     : [50, 1998],                                                                            
    "Pretending to pick something up"                                                                       : [50, 1969],                                                                        
    "Pretending to open something without actually opening it"                                              : [50, 1911],                                                                                                
    "Pulling something from left to right"                                                                  : [50, 1908],                                                                            
    "Lifting something up completely without letting it drop down"                                          : [50, 1906],                                                                                                    
    "Holding something next to something"                                                                   : [50, 1893],                                                                            
    "Pulling something from right to left"                                                                  : [50, 1886],                                                                            
    "Opening something"                                                                                     : [50, 1869],                                                            
    "Something falling like a feather or paper"                                                             : [50, 1858],                                                                                    
    "Lifting something up completely, then letting it drop down"                                            : [50, 1851],                                                                                                    
    "Holding something"                                                                                     : [50, 1851],                                                            
    "Putting something onto something"                                                                      : [50, 1850],                                                                        
    "Lifting up one end of something, then letting it drop down"                                            : [50, 1850],                                                                                                    
    "Pushing something with something"                                                                      : [50, 1804],                                                                        
    "Holding something over something"                                                                      : [50, 1804],                                                                        
    "Rolling something on a flat surface"                                                                   : [50, 1773],                                                                            
    "Touching (without moving) part of something"                                                           : [50, 1763],                                                                                    
    "Pretending to put something on a surface"                                                              : [50, 1644],                                                                                
    "Dropping something onto something"                                                                     : [50, 1623],                                                                            
    "Lifting up one end of something without letting it drop down"                                          : [50, 1613],                                                                                                    
    "Poking something so it slightly moves"                                                                 : [50, 1599],                                                                                
    "Spinning something that quickly stops spinning"                                                        : [50, 1587],                                                                                        
    "Showing that something is inside something"                                                            : [50, 1547],                                                                                    
    "Folding something"                                                                                     : [50, 1542],                                                            
    "Pouring something into something"                                                                      : [50, 1530],                                                                        
    "Closing something"                                                                                     : [50, 1482],                                                            
    "Throwing something against something"                                                                  : [50, 1475],                                                                            
    "Stacking number of something"                                                                          : [50, 1463],                                                                    
    "Picking something up"                                                                                  : [50, 1456],                                                            
    "Pretending to take something from somewhere"                                                           : [50, 1437],                                                                                    
    "Putting something behind something"                                                                    : [50, 1428],                                                                            
    "Moving something closer to something"                                                                  : [50, 1426],                                                                            
    "Holding something behind something"                                                                    : [50, 1374],                                                                            
    "Putting something and something on the table"                                                          : [50, 1353],                                                                                    
    "Moving something away from something"                                                                  : [50, 1352],                                                                            
    "Approaching something with your camera"                                                                : [50, 1349],                                                                                
    "Pushing something so that it almost falls off but doesn't"                                             : [50, 1321],                                                                                                    
    "Showing something on top of something"                                                                 : [50, 1301],                                                                                
    "Pretending to put something next to something"                                                         : [50, 1297],                                                                                        
    "Taking something from somewhere"                                                                       : [50, 1290],                                                                        
    "Tilting something with something on it until it falls off"                                             : [50, 1272],                                                                                                    
    "Unfolding something"                                                                                   : [50, 1266],                                                            
    "Pretending to be tearing something that is not tearable"                                               : [50, 1256],                                                                                                
    "Turning the camera left while filming something"                                                       : [50, 1239],                                                                                        
    "Turning the camera right while filming something"                                                      : [50, 1239],                                                                                        
    "Dropping something next to something"                                                                  : [50, 1232],                                                                            
    "Attaching something to something"                                                                      : [50, 1227],                                                                        
    "Dropping something into something"                                                                     : [50, 1222],                                                                            
    "Putting something, something and something on the table"                                               : [50, 1211],                                                                                                
    "Moving away from something with your camera"                                                           : [50, 1199],                                                                                    
    "Showing something next to something"                                                                   : [50, 1185],                                                                            
    "Putting number of something onto something"                                                            : [50, 1180],                                                                                    
    "Throwing something in the air and catching it"                                                         : [50, 1177],                                                                                        
    "Plugging something into something but pulling it right out as you remove your hand"                    : [50, 1176],                                                                                                                            
    "Spinning something so it continues spinning"                                                           : [50, 1168],                                                                                    
    "Pretending to put something into something"                                                            : [50, 1165],                                                                                    
    "Letting something roll along a flat surface"                                                           : [50, 1163],                                                                                    
    "Piling something up"                                                                                   : [50, 1145],                                                            
    "Twisting something"                                                                                    : [50, 1131],                                                            
    "Dropping something in front of something"                                                              : [50, 1131],                                                                                
    "Scooping something up with something"                                                                  : [50, 1123],                                                                            
    "Pretending to close something without actually closing it"                                             : [50, 1122],                                                                                                    
    "Putting something in front of something"                                                               : [50, 1094],                                                                                
    "Removing something, revealing something behind"                                                        : [50, 1069],                                                                                        
    "Showing something to the camera"                                                                       : [50, 1061],                                                                        
    "Pretending to take something out of something"                                                         : [50, 1045],                                                                                        
    "Throwing something in the air and letting it fall"                                                     : [50, 1038],                                                                                            
    "Throwing something onto a surface"                                                                     : [50, 1035],                                                                            
    "Turning the camera upwards while filming something"                                                    : [50, 1021],                                                                                            
    "Pretending to throw something"                                                                         : [50, 1019],                                                                        
    "Moving something towards the camera"                                                                   : [50, 994 ],                                                                            
    "Trying to bend something unbendable so nothing happens"                                                : [50, 991 ],                                                                                                
    "Dropping something behind something"                                                                   : [50, 991 ],                                                                            
    "Moving something away from the camera"                                                                 : [50, 986 ],                                                                            
    "Putting something upright on the table"                                                                : [50, 980 ],                                                                                
    "Turning the camera downwards while filming something"                                                  : [50, 976 ],                                                                                            
    "Laying something on the table on its side, not upright"                                                : [50, 950 ],                                                                                                
    "Showing a photo of something to the camera"                                                            : [50, 916 ],                                                                                    
    "Moving part of something"                                                                              : [50, 905 ],                                                                
    "Tipping something over"                                                                                : [50, 896 ],                                                                
    "Poking something so that it falls over"                                                                : [50, 892 ],                                                                                
    "Pretending to turn something upside down"                                                              : [50, 888 ],                                                                                
    "Moving something across a surface until it falls down"                                                 : [50, 883 ],                                                                                            
    "Letting something roll down a slanted surface"                                                         : [50, 876 ],                                                                                    
    "Wiping something off of something"                                                                     : [50, 873 ],                                                                        
    "Pretending to squeeze something"                                                                       : [50, 856 ],                                                                        
    "Pushing something so it spins"                                                                         : [50, 845 ],                                                                    
    "Putting something that cannot actually stand upright upright on the table, so it falls on its side"    : [50, 837 ],                                                                                                                                            
    "Moving something across a surface without it falling down"                                             : [50, 832 ],                                                                                                
    "Tilting something with something on it slightly so it doesn't fall down"                               : [50, 829 ],                                                                                                                
    "Bending something so that it deforms"                                                                  : [50, 798 ],                                                                            
    "Pretending to poke something"                                                                          : [50, 754 ],                                                                    
    "Putting something underneath something"                                                                : [50, 748 ],                                                                                
    "Pretending to put something behind something"                                                          : [50, 746 ],                                                                                    
    "Pretending to put something onto something"                                                            : [50, 740 ],                                                                                    
    "Pulling something out of something"                                                                    : [50, 736 ],                                                                            
    "Bending something until it breaks"                                                                     : [50, 718 ],                                                                        
    "Pushing something off of something"                                                                    : [50, 687 ],                                                                            
    "Burying something in something"                                                                        : [50, 687 ],                                                                        
    "Trying but failing to attach something to something because it doesn't stick"                          : [50, 660 ],                                                                                                                    
    "Something colliding with something and both are being deflected"                                       : [50, 653 ],                                                                                                        
    "Pulling two ends of something but nothing happens"                                                     : [50, 643 ],                                                                                        
    "Putting something on the edge of something so it is not supported and falls down"                      : [50, 638 ],                                                                                                                        
    "Pulling something from behind of something"                                                            : [50, 586 ],                                                                                    
    "Moving something and something so they pass each other"                                                : [50, 582 ],                                                                                                
    "Moving something and something so they collide with each other"                                        : [50, 577 ],                                                                                                        
    "Putting something on a flat surface without letting it roll"                                           : [50, 553 ],                                                                                                    
    "Something colliding with something and both come to a halt"                                            : [50, 547 ],                                                                                                    
    "Pretending to sprinkle air onto something"                                                             : [50, 543 ],                                                                                
    "Sprinkling something onto something"                                                                   : [50, 540 ],                                                                            
    "Spreading something onto something"                                                                    : [50, 535 ],                                                                            
    "Digging something out of something"                                                                    : [50, 522 ],                                                                            
    "Pouring something out of something"                                                                    : [50, 514 ],                                                                            
    "Something being deflected from something"                                                              : [50, 492 ],                                                                                
    "Pretending or failing to wipe something off of something"                                              : [50, 490 ],                                                                                                
    "Spilling something onto something"                                                                     : [50, 474 ],                                                                        
    "Tipping something with something in it over, so something in it falls out"                             : [50, 447 ],                                                                                                                
    "Putting something that can't roll onto a slanted surface, so it stays where it is"                     : [50, 447 ],                                                                                                                        
    "Pretending to pour something out of something, but something is empty"                                 : [50, 445 ],                                                                                                            
    "Putting something that can't roll onto a slanted surface, so it slides down"                           : [50, 442 ],                                                                                                                    
    "Putting something onto something else that cannot support it so it falls down"                         : [50, 442 ],                                                                                                                    
    "Letting something roll up a slanted surface, so it rolls back down"                                    : [50, 441 ],                                                                                                            
    "Pulling two ends of something so that it gets stretched"                                               : [50, 438 ],                                                                                                
    "Pushing something onto something"                                                                      : [50, 419 ],                                                                        
    "Twisting (wringing) something wet until water comes out"                                               : [50, 408 ],                                                                                                
    "Lifting a surface with something on it until it starts sliding down"                                   : [50, 405 ],                                                                                                            
    "Pretending or trying and failing to twist something"                                                   : [50, 404 ],                                                                                            
    "Pouring something onto something"                                                                      : [50, 403 ],                                                                        
    "Pretending to scoop something up with something"                                                       : [50, 389 ],                                                                                        
    "Pretending to put something underneath something"                                                      : [50, 373 ],                                                                                        
    "Poking a stack of something so the stack collapses"                                                    : [50, 367 ],                                                                                            
    "Failing to put something into something because something does not fit"                                : [50, 353 ],                                                                                                                
    "Pouring something into something until it overflows"                                                   : [50, 352 ],                                                                                            
    "Pulling something onto something"                                                                      : [50, 343 ],                                                                        
    "Pulling two ends of something so that it separates into two pieces"                                    : [50, 313 ],                                                                                                            
    "Poking a stack of something without the stack collapsing"                                              : [50, 276 ],                                                                                                
    "Lifting a surface with something on it but not enough for it to slide down"                            : [50, 268 ],                                                                                                                    
    "Trying to pour something into something, but missing so it spills next to it"                          : [50, 265 ],                                                                                                                    
    "Poking a hole into something soft"                                                                     : [50, 258 ],                                                                        
    "Spilling something next to something"                                                                  : [50, 240 ],                                                                            
    "Pretending to spread air onto something"                                                               : [50, 225 ],                                                                                
    "Poking something so that it spins around"                                                              : [50, 185 ],                                                                                
    "Putting something onto a slanted surface but it doesn't glide down"                                    : [50, 183 ],                                                                                                            
    "Spilling something behind something"                                                                   : [50, 143 ],                                                                            
    "Poking a hole into some substance"                                                                     : [50, 115 ],                                                                        
}

assert __LABELS__.keys() == __SAMPLES_PER_LABEL__.keys(), \
    "Inconsistent"

# add UNKNOWN LABEL
__LABELS__[UNKOWN_LABEL] = UNKOWN_CID
__SAMPLES_PER_LABEL__[UNKOWN_LABEL] = [0, 999999]


# ------------------------------------------------------------------------ #
#                 Collect Annotations for Each Sample                      #
# ------------------------------------------------------------------------ #

# NOTE: __ANNOTATIONS__ is a Python key word
# Currently, this dataset only provides annotation for training & validation
# We use None to mark unlabelled samples
__ANNOTATIONS__ = dict()

ANNOT_FILE = os.path.join(DIR_PATH, "something-something-v2.annot")
if (os.path.exists(ANNOT_FILE)
        and (touch_date(FILE_PATH) < touch_date(ANNOT_FILE))
   ):
    logger.info("Find valid annotation cache")
    fin = open(ANNOT_FILE, "rb")
    __ANNOTATIONS__ = pickle.load(fin)
    fin.close()
else:
    logger.info("Building annotation data...")
    ## training/validation set has labels
    for _jlist in (TRAINSET_JLIST, VALSET_JLIST):
        for _jdict in _jlist:
            video = str(_jdict["id"])
            label = str(_jdict["template"]).replace('[', '').replace(']', '')
            __ANNOTATIONS__[video] = label
    ## testing set doesn't have labels
    for _jlist in (TESTSET_JLIST, ):
        for _jdict in _jlist:
            video = str(_jdict["id"])
            __ANNOTATIONS__[video] = UNKOWN_LABEL  
    ## TODO: write failure check
    f = open(ANNOT_FILE, "wb")
    pickle.dump(__ANNOTATIONS__, f)
    f.close()


def test():
    """Self-testing function
    """
    print(len(__ANNOTATIONS__))
    print(sorted(CIDS))

    sample_num = 0
    for _label in __SAMPLES_PER_LABEL__:
        sample_num += __SAMPLES_PER_LABEL__[_label][1]
    print(sample_num)

if __name__ == "__main__":
    test()