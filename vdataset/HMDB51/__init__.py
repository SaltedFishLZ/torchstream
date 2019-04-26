# HMDB51 Dataset

__classes__ = [
    "brush_hair"    ,
    "cartwheel"     ,
    "catch"         ,
    "chew"          ,
    "clap"          ,
    "climb"         ,
    "climb_stairs"  ,
    "dive"          ,
    "draw_sword"    ,
    "dribble"       ,
    "drink"         ,
    "eat"           ,
    "fall_floor"    ,
    "fencing"       ,
    "flic_flac"     ,
    "golf"          ,
    "handstand"     ,
    "hit"           ,
    "hug"           ,
    "jump"          ,
    "kick"          ,
    "kick_ball"     ,
    "kiss"          ,
    "laugh"         ,
    "pick"          ,
    "pour"          ,
    "pullup"        ,
    "punch"         ,
    "push"          ,
    "pushup"        ,
    "ride_bike"     ,
    "ride_horse"    ,
    "run"           ,
    "shake_hands"   ,
    "shoot_ball"    ,
    "shoot_bow"     ,
    "shoot_gun"     ,
    "sit"           ,
    "situp"         ,
    "smile"         ,
    "smoke"         ,
    "somersault"    ,
    "stand"         ,
    "swing_baseball",
    "sword"         ,
    "sword_exercise",
    "talk"          ,
    "throw"         ,
    "turn"          ,
    "walk"          ,
    "wave"          ,
]

# From the paper we can know each class has at least 101 samples
# Page 2 of the paper
# paper link:
# http://serre-lab.clps.brown.edu/wp-content/uploads/2012/08/Kuehne_etal_iccv11.pdf
# So we estimate it as [101, int(2**31)]
__samples__ = dict(zip(__classes__, 51*[[101, int(2**31)]]))

from .data_path import raw_data_path, prc_data_path
from .label_map import label_map
from .data_split import for_train, for_val, for_test

__all__ = [
    "__classes__", "__samples__",
    "raw_data_path", "prc_data_path",
    "label_map", "for_train", "for_val", "for_test"
]

if __name__ == "__main__":
    print("Common Data of HMDB51 Dataset")
    print(__samples__)
