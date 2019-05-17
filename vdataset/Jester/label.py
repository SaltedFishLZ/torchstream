
from .common import trainset_df, valset_df, testset_df


# ------------------------------------------------------------------------ #
#                   Labels and Corresponding CIDs                          #
# ------------------------------------------------------------------------ #

__classes__ = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Down",
    "Swiping Up",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Pushing Two Fingers Away",
    "Pulling Two Fingers In",
    "Rolling Hand Forward",
    "Rolling Hand Backward",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Thumb Up",
    "Thumb Down",
    "Shaking Hand",
    "Stop Sign",
    "Drumming Fingers",
    "No gesture",
    "Doing other things",
]

# generating label-cid mapping
# map "Doing other things" cid 0
cids = list(range(len(__classes__)))
cids = cids[1:-1] + [0]
__labels__ = dict(zip(__classes__, cids))



__sample_num_per_class__ = {
    "Doing other things"                :12416  ,
    "Drumming Fingers"                  :5444   ,
    "No gesture"                        :5344   ,
    "Pulling Hand In"                   :5379   ,
    "Pulling Two Fingers In"            :5315   ,
    "Pushing Hand Away"                 :5434   ,
    "Pushing Two Fingers Away"          :5358   ,
    "Rolling Hand Backward"             :5031   ,
    "Rolling Hand Forward"              :5165   ,
    "Shaking Hand"                      :5314   ,
    "Sliding Two Fingers Down"          :5410   ,
    "Sliding Two Fingers Left"          :5345   ,
    "Sliding Two Fingers Right"         :5244   ,
    "Sliding Two Fingers Up"            :5262   ,
    "Stop Sign"                         :5413   ,
    "Swiping Down"                      :5303   ,
    "Swiping Left"                      :5160   ,
    "Swiping Right"                     :5066   ,
    "Swiping Up"                        :5240   ,
    "Thumb Down"                        :5460   ,
    "Thumb Up"                          :5457   ,
    "Turning Hand Clockwise"            :3980   ,
    "Turning Hand Counterclockwise"     :4181   ,
    "Zooming In With Full Hand"         :5307   ,
    "Zooming In With Two Fingers"       :5355   ,
    "Zooming Out With Full Hand"        :5330   ,
    "Zooming Out With Two Fingers"      :5379   ,
}



# ------------------------------------------------------------------------ #
#                 Collect Annotations for Each Sample                      #
# ------------------------------------------------------------------------ #

# NOTE:
# __annotations__ is a Python key word
# So, we use __targets__
__targets__ = dict()

# Currently, this dataset only provides annotation for training & validation
for df in [trainset_df, valset_df]:
    for idx, row in df.iterrows():
        video = row["video"]
        label = row["label"]
        __targets__[video] = label




if __name__ == "__main__":    
    print(len(__targets__))
    print(cids)