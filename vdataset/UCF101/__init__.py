# UCF101 Dataset
# https://www.crcv.ucf.edu/data/UCF101.php

__classes__ = [
    "ApplyEyeMakeup"    ,
    "ApplyLipstick"     ,
    "Archery"           ,
    "BabyCrawling"      ,
    "BalanceBeam"       ,
    "BandMarching"      ,
    "BaseballPitch"     ,
    "Basketball"        ,
    "BasketballDunk"    ,
    "BenchPress"        ,
    "Biking"            ,
    "Billiards"         ,
    "BlowDryHair"       ,
    "BlowingCandles"    ,
    "BodyWeightSquats"  ,
    "Bowling"           ,
    "BoxingPunchingBag" ,
    "BoxingSpeedBag"    ,
    "BreastStroke"      ,
    "BrushingTeeth"     ,
    "CleanAndJerk"      ,
    "CliffDiving"       ,
    "CricketBowling"    ,
    "CricketShot"       ,
    "CuttingInKitchen"  ,
    "Diving"            ,
    "Drumming"          ,
    "Fencing"           ,
    "FieldHockeyPenalty",
    "FloorGymnastics"   ,
    "FrisbeeCatch"      ,
    "FrontCrawl"        ,
    "GolfSwing"         ,
    "Haircut"           ,
    "Hammering"         ,
    "HammerThrow"       ,
    "HandstandPushups"  ,
    "HandstandWalking"  ,
    "HeadMassage"       ,
    "HighJump"          ,
    "HorseRace"         ,
    "HorseRiding"       ,
    "HulaHoop"          ,
    "IceDancing"        ,
    "JavelinThrow"      ,
    "JugglingBalls"     ,
    "JumpingJack"       ,
    "JumpRope"          ,
    "Kayaking"          ,
    "Knitting"          ,
    "LongJump"          ,
    "Lunges"            ,
    "MilitaryParade"    ,
    "Mixing"            ,
    "MoppingFloor"      ,
    "Nunchucks"         ,
    "ParallelBars"      ,
    "PizzaTossing"      ,
    "PlayingCello"      ,
    "PlayingDaf"        ,
    "PlayingDhol"       ,
    "PlayingFlute"      ,
    "PlayingGuitar"     ,
    "PlayingPiano"      ,
    "PlayingSitar"      ,
    "PlayingTabla"      ,
    "PlayingViolin"     ,
    "PoleVault"         ,
    "PommelHorse"       ,
    "PullUps"           ,
    "Punch"             ,
    "PushUps"           ,
    "Rafting"           ,
    "RockClimbingIndoor",
    "RopeClimbing"      ,
    "Rowing"            ,
    "SalsaSpin"         ,
    "ShavingBeard"      ,
    "Shotput"           ,
    "SkateBoarding"     ,
    "Skiing"            ,
    "Skijet"            ,
    "SkyDiving"         ,
    "SoccerJuggling"    ,
    "SoccerPenalty"     ,
    "StillRings"        ,
    "SumoWrestling"     ,
    "Surfing"           ,
    "Swing"             ,
    "TableTennisShot"   ,
    "TaiChi"            ,
    "TennisSwing"       ,
    "ThrowDiscus"       ,
    "TrampolineJumping" ,
    "Typing"            ,
    "UnevenBars"        ,
    "VolleyballSpiking" ,
    "WalkingWithDog"    ,
    "WallPushups"       ,
    "WritingOnBoard"    ,
    "YoYo"              ,
]

# Details from UCF's official website
# https://www.crcv.ucf.edu/data/UCF101.php
# "The videos in 101 action categories are grouped into 25 groups, where 
# each group can consist of 4-7 videos of an action. The videos from the
# same group may share some common features, such as similar background,
# similar viewpoint, etc."
# So, we choose to use [25 * 4, 25 * 7] as the sample num's [min, max] for
# each class
__samples__ = dict(zip(__classes__, 101*[[25*4, 25*7]]))

__all__ = ["raw_data_path", "prc_data_path", "label_map",
        "__supported_splits__", "for_train", "for_val", "for_test"]

from .data_path import raw_data_path, prc_data_path
from .label_map import label_map
from .data_split import for_train, for_val, for_test

__all__ = [
    "__classes__", "__samples__",
    "raw_data_path", "prc_data_path",
    "label_map", "for_train", "for_val", "for_test"
]

if __name__ == "__main__":
    print("common Data of Weizmann Dataset")
    print(__samples__)
