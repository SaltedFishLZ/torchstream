import importlib

from .dataset import VideoDataset
from .metadata.metasets import jester_v1 as jester_v1

class JesterV1(VideoDataset):

    def __init__(self, train=True, transform=None, target_transform=None,
                 **kwargs):
        """
        """

        mod = "RGB"
        ext = "jpg"
        root = jester_v1.JPG_DATA_PATH
        layout = jester_v1.__layout__
        class_to_idx = jester_v1.__LABELS__
        annots = jester_v1.__ANNOTATIONS__

        if train:
            datapoint_filter = jester_v1.TrainsetFilter()
        else:
            datapoint_filter = jester_v1.TestsetFilter()

        super(JesterV1, self).__init__(root=root, layout=layout,
                                       annots=annots,
                                       class_to_idx=class_to_idx,
                                       mod=mod, ext=ext, tmpl="{0:05d}", offset=1,
                                       datapoint_filter=datapoint_filter,
                                       transform=transform,
                                       target_transform=target_transform,
                                       **kwargs)
