# Jester V1 Dataset
# https://20bn.com/datasets/jester/v1

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

# generating label mapping
# map "Doing other things" cid 0
cids = list(range(len(__classes__)))
cids = cids[1:-1] + [0]
label_map = dict(zip(__classes__, cids))


if __name__ == "__main__":
    print(cids)