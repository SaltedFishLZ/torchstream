import importlib

from .dataset import VideoDataset
from .metadata.metasets import ucf101 as ucf101

class UCF101(VideoDataset):

    def __init__(self, train=True, transform=None, target_transform=None,
                 **kwargs):
        """
        """

        mod = "RGB"
        ext = "avi"
        root = ucf101.AVI_DATA_PATH
        layout = ucf101.__layout__
        class_to_idx = ucf101.__LABELS__

        if train:
            datapoint_filter = ucf101.TrainsetFilter()
        else:
            datapoint_filter = ucf101.TestsetFilter()

        super(UCF101, self).__init__(root=root, layout=layout,
                                     class_to_idx=class_to_idx,
                                     mod=mod, ext=ext,
                                     datapoint_filter=datapoint_filter,
                                     transform=transform,
                                     target_transform=target_transform,
                                     **kwargs
                                    )
