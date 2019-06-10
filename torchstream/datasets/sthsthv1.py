import importlib

from .dataset import VideoDataset
from .metadata.metasets import sth_sth_v1 as sth_sth_v1

class SomethingSomethingV1(VideoDataset):

    def __init__(self, train=True, transform=None, target_transform=None,
                 **kwargs):
        """
        """

        mod = "RGB"
        ext = "avi"
        root = sth_sth_v1.AVI_DATA_PATH
        layout = sth_sth_v1.__layout__
        class_to_idx = sth_sth_v1.__LABELS__
        annots = sth_sth_v1.__ANNOTATIONS__

        if train:
            datapoint_filter = sth_sth_v1.TrainsetFilter()
        else:
            # here, we use validation set
            datapoint_filter = sth_sth_v1.ValsetFilter()

        super(SomethingSomethingV1, self).__init__(root=root, layout=layout,
                                                   annots=annots,
                                                   class_to_idx=class_to_idx,
                                                   mod=mod, ext=ext, tmpl="{0:05d}", offset=1,
                                                   datapoint_filter=datapoint_filter,
                                                   transform=transform,
                                                   target_transform=target_transform,
                                                   **kwargs)
