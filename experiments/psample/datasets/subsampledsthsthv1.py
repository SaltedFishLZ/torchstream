import numpy as np
import multiprocessing as mp

from torchstream.datasets.dataset import VideoDataset
from torchstream.datasets.metadata.metasets import sth_sth_v1 as sth_sth_v1

from torchstream.datasets.imgseq import ImageSequence, _to_imgseq
from torchstream.datasets.vidarr import VideoArray,_to_vidarr

class SubsampledSomethingSomethingV1(VideoDataset):

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

        super(SubsampledSomethingSomethingV1, self).__init__(root=root, layout=layout,
                                                   annots=annots,
                                                   class_to_idx=class_to_idx,
                                                   mod=mod, ext=ext, tmpl="{0:05d}", offset=1,
                                                   datapoint_filter=datapoint_filter,
                                                   transform=transform,
                                                   target_transform=target_transform,
                                                   **kwargs)


        print("Original Number", len(self.samples))

        # subsample datapoints
        N = len(self.datapoints)
        new_datapoints = []
        for i in range(N):
            if i % 10 == 0:
                new_datapoints.append(self.datapoints[i])
        self.datapoints = new_datapoints

        p = mp.Pool(32)
        if self.seq:
            self.samples = p.map(_to_imgseq, self.datapoints)
        else:
            self.samples = p.map(_to_vidarr, self.datapoints)

        print("Subsampled Number", len(self.samples))
