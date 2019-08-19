"""Annotation data
"""
__all__ = [
    "__LABELS__", "__SAMPLES_PER_LABEL__",
    "__ANNOTATIONS__"
]

import os
import pickle
import logging

from . import __config__
from .csvparse import TRAINSET_DF, VALSET_DF, TESTSET_DF
from ...__const__ import UNKNOWN_LABEL, UNKNOWN_CID
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
#                   Labels and Corresponding CIDS                          #
# ------------------------------------------------------------------------ #

__SAMPLES_PER_LABEL__ = {
    "Holding something"                                                                                     :   [10, 986] ,                                            
    "Turning something upside down"                                                                         :   [10, 979] ,
    "Turning the camera left while filming something"                                                       :   [10, 924] ,                                                
    "Stacking number of something"                                                                          :   [10, 914] ,                            
    "Turning the camera right while filming something"                                                      :   [10, 914] ,                                        
    "Opening something"                                                                                     :   [10, 888] ,      
    "Approaching something with your camera"                                                                :   [10, 885] , 
    "Picking something up"                                                                                  :   [10, 877] ,        
    "Pushing something so that it almost falls off but doesn't"                                             :   [10, 873] ,        
    "Folding something"                                                                                     :   [10, 864] ,      
    "Moving something away from the camera"                                                                 :   [10, 863] ,          
    "Closing something"                                                                                     :   [10, 858] ,      
    "Moving away from something with your camera"                                                           :   [10, 850] ,      
    "Turning the camera downwards while filming something"                                                  :   [10, 845] ,      
    "Pushing something so that it slightly moves"                                                           :   [10, 841] ,      
    "Turning the camera upwards while filming something"                                                    :   [10, 839] ,          
    "Pretending to pick something up"                                                                       :   [10, 838] ,                  
    "Showing something to the camera"                                                                       :   [10, 838] ,                           
    "Moving something up"                                                                                   :   [10, 833] ,                  
    "Plugging something into something"                                                                     :   [10, 830] ,                  
    "Unfolding something"                                                                                   :   [10, 830] ,          
    "Putting something onto something"                                                                      :   [10, 828] ,              
    "Showing that something is empty"                                                                       :   [10, 827] ,        
    "Pretending to put something on a surface"                                                              :   [10, 825] ,       
    "Taking something from somewhere"                                                                       :   [10, 825] ,    
    "Putting something next to something"                                                                   :   [10, 824] ,
    "Moving something towards the camera"                                                                   :   [10, 821] ,
    "Showing a photo of something to the camera"                                                            :   [10, 820] ,                                                 
    "Pushing something with something"                                                                      :   [10, 815] ,                                
    "Throwing something"                                                                                    :   [10, 808] ,                                    
    "Pushing something from left to right"                                                                  :   [10, 802] ,                                        
    "Something falling like a feather or paper"                                                             :   [10, 801] ,                                
    "Throwing something in the air and letting it fall"                                                     :   [10, 801] ,                                                
    "Throwing something against something"                                                                  :   [10, 796] ,                                            
    "Lifting something with something on it"                                                                :   [10, 793] ,                            
    "Taking one of many similar things on the table"                                                        :   [10, 788] ,                                                
    "Showing something behind something"                                                                    :   [10, 785] ,                                
    "Putting something into something"                                                                      :   [10, 781] ,                                        
    "Tearing something just a little bit"                                                                   :   [10, 780] ,                                    
    "Moving something away from something"                                                                  :   [10, 779] ,                        
    "Tearing something into two pieces"                                                                     :   [10, 778] ,                                                    
    "Holding something next to something"                                                                   :   [10, 777] ,                                                
    "Pushing something from right to left"                                                                  :   [10, 777] ,                                                            
    "Putting something, something and something on the table"                                               :   [10, 776] ,                                    
    "Moving something closer to something"                                                                  :   [10, 775] ,                                        
    "Pretending to take something from somewhere"                                                           :   [10, 775] ,                                            
    "Pretending to put something next to something"                                                         :   [10, 774] ,                                                    
    "Uncovering something"                                                                                  :   [10, 773] ,                                        
    "Pouring something into something"                                                                      :   [10, 772] ,                                            
    "Putting something and something on the table"                                                          :   [10, 772] ,                                        
    "Something falling like a rock"                                                                         :   [10, 772] ,                                    
    "Moving something down"                                                                                 :   [10, 769] ,                            
    "Pulling something from right to left"                                                                  :   [10, 769] ,                                        
    "Throwing something in the air and catching it"                                                         :   [10, 767] ,                            
    "Tilting something with something on it until it falls off"                                             :   [10, 763] ,                                
    "Putting something in front of something"                                                               :   [10, 762] ,                                    
    "Pretending to turn something upside down"                                                              :   [10, 760] ,                                            
    "Putting something on a surface"                                                                        :   [10, 759] ,                                        
    "Pretending to throw something"                                                                         :   [10, 757] ,                                        
    "Covering something with something"                                                                     :   [10, 756] ,                                        
    "Showing something on top of something"                                                                 :   [10, 756] ,                                            
    "Squeezing something"                                                                                   :   [10, 753] ,                                    
    "Putting something similar to other things that are already on the table"                               :   [10, 752] ,          
    "Lifting up one end of something, then letting it drop down"                                            :   [10, 751] ,                  
    "Taking something out of something"                                                                     :   [10, 749] ,  
    "Moving part of something"                                                                              :   [10, 747] ,                          
    "Pulling something from left to right"                                                                  :   [10, 745] ,                          
    "Lifting something up completely without letting it drop down"                                          :   [10, 744] ,                          
    "Attaching something to something"                                                                      :   [10, 743] ,                  
    "Holding something in front of something"                                                               :   [10, 743] ,          
    "Moving something and something closer to each other"                                                   :   [10, 743] ,                      
    "Putting something behind something"                                                                    :   [10, 743] ,              
    "Pushing something so that it falls off the table"                                                      :   [10, 742] ,                              
    "Holding something over something"                                                                      :   [10, 735] ,          
    "Pretending to open something without actually opening it"                                              :   [10, 734] ,                      
    "Removing something, revealing something behind"                                                        :   [10, 732] ,                          
    "Hitting something with something"                                                                      :   [10, 729] ,                  
    "Moving something and something away from each other"                                                   :   [10, 727] ,                  
    "Touching (without moving) part of something"                                                           :   [10, 727] ,          
    "Pretending to put something into something"                                                            :   [10, 724] ,                              
    "Showing that something is inside something"                                                            :   [10, 724] ,                                          
    "Lifting something up completely, then letting it drop down"                                            :   [10, 721] ,                          
    "Pretending to take something out of something"                                                         :   [10, 720] ,                                              
    "Holding something behind something"                                                                    :   [10, 709] ,                                              
    "Laying something on the table on its side, not upright"                                                :   [10, 707] ,                                  
    "Poking something so it slightly moves"                                                                 :   [10, 700] ,                                          
    "Pretending to close something without actually closing it"                                             :   [10, 699] ,                                  
    "Putting something upright on the table"                                                                :   [10, 698] ,                                              
    "Dropping something in front of something"                                                              :   [10, 690] ,                                  
    "Dropping something behind something"                                                                   :   [10, 687] ,                                                                          
    "Lifting up one end of something without letting it drop down"                                          :   [10, 685] ,                                          
    "Rolling something on a flat surface"                                                                   :   [10, 682] ,                                      
    "Throwing something onto a surface"                                                                     :   [10, 677] ,                                                  
    "Showing something next to something"                                                                   :   [10, 671] ,                                                      
    "Dropping something onto something"                                                                     :   [10, 668] ,                                  
    "Stuffing something into something"                                                                     :   [10, 668] ,                                      
    "Dropping something into something"                                                                     :   [10, 662] ,                                  
    "Piling something up"                                                                                   :   [10, 662] ,                              
    "Letting something roll along a flat surface"                                                           :   [10, 660] ,                      
    "Twisting something"                                                                                    :   [10, 658] ,                                  
    "Spinning something that quickly stops spinning"                                                        :   [10, 643] ,                                  
    "Putting number of something onto something"                                                            :   [10, 636] ,                                      
    "Moving something across a surface without it falling down"                                             :   [10, 634] ,                                      
    "Putting something underneath something"                                                                :   [10, 634] ,                          
    "Plugging something into something but pulling it right out as you remove your hand"                    :   [10, 628] ,                                  
    "Dropping something next to something"                                                                  :   [10, 627] ,                              
    "Poking something so that it falls over"                                                                :   [10, 606] ,                                  
    "Spinning something so it continues spinning"                                                           :   [10, 593] ,                              
    "Poking something so lightly that it doesn't or almost doesn't move"                                    :   [10, 588] ,                                          
    "Wiping something off of something"                                                                     :   [10, 585] ,                                  
    "Moving something across a surface until it falls down"                                                 :   [10, 582] ,                                              
    "Pretending to poke something"                                                                          :   [10, 580] ,                                                          
    "Putting something that cannot actually stand upright upright on the table, so it falls on its side"    :   [10, 570] ,                                                            
    "Pulling something out of something"                                                                    :   [10, 566] ,                                
    "Scooping something up with something"                                                                  :   [10, 565] ,                    
    "Pretending to be tearing something that is not tearable"                                               :   [10, 562] ,                                                
    "Burying something in something"                                                                        :   [10, 543] ,                                        
    "Tipping something over"                                                                                :   [10, 542] ,                                            
    "Tilting something with something on it slightly so it doesn't fall down"                               :   [10, 533] ,                                            
    "Pretending to put something onto something"                                                            :   [10, 528] ,                                            
    "Bending something until it breaks"                                                                     :   [10, 522] ,                                                    
    "Letting something roll down a slanted surface"                                                         :   [10, 512] ,                                                    
    "Trying to bend something unbendable so nothing happens"                                                :   [10, 509] ,                                        
    "Bending something so that it deforms"                                                                  :   [10, 505] ,                                            
    "Digging something out of something"                                                                    :   [10, 503] ,                                        
    "Pretending to put something underneath something"                                                      :   [10, 502] ,                                
    "Putting something on a flat surface without letting it roll"                                           :   [10, 497] ,                                            
    "Putting something on the edge of something so it is not supported and falls down"                      :   [10, 479] ,                                
    "Pretending to put something behind something"                                                          :   [10, 471] ,                                        
    "Spreading something onto something"                                                                    :   [10, 471] ,                
    "Sprinkling something onto something"                                                                   :   [10, 466] ,                                            
    "Something colliding with something and both come to a halt"                                            :   [10, 463] ,                                    
    "Pushing something off of something"                                                                    :   [10, 462] ,                                        
    "Putting something that can't roll onto a slanted surface, so it stays where it is"                     :   [10, 453] ,                                                            
    "Lifting a surface with something on it until it starts sliding down"                                   :   [10, 451] ,                                        
    "Pretending or failing to wipe something off of something"                                              :   [10, 433] ,                                                                
    "Trying but failing to attach something to something because it doesn't stick"                          :   [10, 433] ,                                                
    "Pulling something from behind of something"                                                            :   [10, 427] ,                                    
    "Pushing something so it spins"                                                                         :   [10, 423] ,                                        
    "Pouring something onto something"                                                                      :   [10, 420] ,                                
    "Pulling two ends of something but nothing happens"                                                     :   [10, 416] ,                            
    "Moving something and something so they pass each other"                                                :   [10, 413] ,                                                
    "Pretending to sprinkle air onto something"                                                             :   [10, 413] ,                                            
    "Putting something that can't roll onto a slanted surface, so it slides down"                           :   [10, 405] ,                                                                
    "Something colliding with something and both are being deflected"                                       :   [10, 395] ,                            
    "Pretending to squeeze something"                                                                       :   [10, 386] ,
    "Pulling something onto something"                                                                      :   [10, 367] ,                                    
    "Putting something onto something else that cannot support it so it falls down"                         :   [10, 362] ,                                                        
    "Lifting a surface with something on it but not enough for it to slide down"                            :   [10, 358] ,                                        
    "Pouring something out of something"                                                                    :   [10, 358] ,                                        
    "Moving something and something so they collide with each other"                                        :   [10, 346] ,                                        
    "Tipping something with something in it over, so something in it falls out"                             :   [10, 341] ,                                                
    "Letting something roll up a slanted surface, so it rolls back down"                                    :   [10, 339] ,                                        
    "Pretending to scoop something up with something"                                                       :   [10, 318] ,                                            
    "Pretending to pour something out of something, but something is empty"                                 :   [10, 311] ,                                                    
    "Pulling two ends of something so that it gets stretched"                                               :   [10, 294] ,                            
    "Failing to put something into something because something does not fit"                                :   [10, 290] ,                            
    "Pretending or trying and failing to twist something"                                                   :   [10, 288] ,                    
    "Trying to pour something into something, but missing so it spills next to it"                          :   [10, 282] ,                    
    "Something being deflected from something"                                                              :   [10, 277] ,                
    "Poking a stack of something so the stack collapses"                                                    :   [10, 273] ,                                                
    "Spilling something onto something"                                                                     :   [10, 267] ,                            
    "Pulling two ends of something so that it separates into two pieces"                                    :   [10, 245] ,                            
    "Pouring something into something until it overflows"                                                   :   [10, 229] ,                    
    "Pretending to spread air onto something"                                                               :   [10, 220] ,                                    
    "Twisting (wringing) something wet until water comes out"                                               :   [10, 219] ,                                    
    "Poking a hole into something soft"                                                                     :   [10, 217] ,            
    "Spilling something next to something"                                                                  :   [10, 207] ,                                        
    "Poking a stack of something without the stack collapsing"                                              :   [10, 206] ,                                            
    "Putting something onto a slanted surface but it doesn't glide down"                                    :   [10, 183] ,                                                                
    "Pushing something onto something"                                                                      :   [10, 170] ,                                                    
    "Poking something so that it spins around"                                                              :   [10, 141] ,                                                
    "Spilling something behind something"                                                                   :   [10, 121] ,                                            
    "Poking a hole into some substance"                                                                     :   [10, 77 ] ,       
}

__LABELS__ = sorted(list(__SAMPLES_PER_LABEL__.keys()))

# generating label-cid mapping
CIDS = list(range(len(__LABELS__)))
__LABELS__ = dict(zip(__LABELS__, CIDS))

# add UNKNOWN LABEL
__LABELS__[UNKNOWN_LABEL] = UNKNOWN_CID
__SAMPLES_PER_LABEL__[UNKNOWN_LABEL] = [0, 999999]



# ------------------------------------------------------------------------ #
#                 Collect Annotations for Each Sample                      #
# ------------------------------------------------------------------------ #

# NOTE: __annotations__ is a Python key word
# Currently, this dataset only provides annotation for training & validation
# We use None to mark unlabelled samples
__ANNOTATIONS__ = dict()

ANNOT_FILE = os.path.join(DIR_PATH, "something-something-v1.annot")
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
    for df in (TRAINSET_DF, VALSET_DF):
        for idx, row in df.iterrows():
            video = str(row["video"])
            label = str(row["label"])
            __ANNOTATIONS__[video] = label
    ## testing set doesn't have labels
    for df in (TESTSET_DF, ):
        for idx, row in df.iterrows():
            video = str(row["video"])
            __ANNOTATIONS__[video] = UNKNOWN_LABEL
    # ## TODO: write failure check
    # fout = open(ANNOT_FILE, "wb")
    # pickle.dump(__ANNOTATIONS__, fout)
    # fout.close()


## Self Test Function
def test():
    """
    Self-testing function
    """
    print(len(__ANNOTATIONS__))
    print(__LABELS__)

if __name__ == "__main__":
    test()
