import importlib

from .dataset import VideoDataset
from .metadata.metasets import hmdb51 as hmdb51

class HMDB51(VideoDataset):

    def __init__(self, train=True, transform=None, target_transform=None,
                 **kwargs):
        """
        """

        mod = "RGB"
        ext = "avi"
        root = hmdb51.AVI_DATA_PATH
        layout = hmdb51.__layout__
        class_to_idx = hmdb51.__LABELS__

        if train:
            datapoint_filter = hmdb51.TrainsetFilter()
        else:
            datapoint_filter = hmdb51.TestsetFilter()
        super(HMDB51, self).__init__(root=root, layout=layout,
                                     class_to_idx=class_to_idx,
                                     mod=mod, ext=ext,
                                     datapoint_filter=datapoint_filter,
                                     transform=transform,
                                     target_transform=target_transform,
                                     **kwargs
                                    )

def test():
    dataset = HMDB51(train=True)
    print(dataset.__len__())

if __name__ == "__main__":
    test()
