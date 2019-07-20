import importlib

from .dataset import VideoDataset
from .metadata.metasets import sth_sth_v2 as sth_sth_v2

class TinySomethingSomethingV2(VideoDataset):

    def __init__(self, train=True, transform=None, target_transform=None,
                 **kwargs):
        """
        """

        mod = "RGB"
        ext = "avi"
        root = sth_sth_v2.AVI_DATA_PATH
        layout = sth_sth_v2.__layout__
        annots = sth_sth_v2.__ANNOTATIONS__


        categories = [
            "Holding something",
            "Turning something upside down",
            "Turning the camera left while filming something",
            "Turning the camera right while filming something",
            "Opening something",
            "Approaching something with your camera",
            "Picking something up",
            "Pretending to pick something up",
            "Pretending to put something onto something",
            "Putting something onto something",
        ]

        class_to_idx = {
            "Holding something"                                 : 0,
            "Turning something upside down"                     : 1,
            "Turning the camera left while filming something"   : 2,
            "Turning the camera right while filming something"  : 3,
            "Opening something"                                 : 4,
            "Approaching something with your camera"            : 5,
            "Picking something up"                              : 6,
            "Pretending to pick something up"                   : 7,
            "Pretending to put something onto something"        : 8,
            "Putting something onto something"                  : 9,
        }


        if train:
            datapoint_filter = sth_sth_v2.TrainsetFilter()
        else:
            # here, we use validation set
            datapoint_filter = sth_sth_v2.ValsetFilter()

        super(TinySomethingSomethingV2, self).__init__(
            root=root, layout=layout,
            annots=annots,
            class_to_idx=class_to_idx,
            mod=mod, ext=ext, tmpl="{0:05d}", offset=1,
            datapoint_filter=datapoint_filter,
            transform=transform,
            target_transform=target_transform,
            **kwargs
            )


        ## filter samples
        print("Filtering datapoints")
        datapoints = []
        for datapoint in self.datapoints:
            if datapoint.label in categories:
                datapoints.append(datapoint)
        self.datapoints = datapoints

        print("Generating samples")
        from torchstream.datasets.imgseq import ImageSequence, _to_imgseq
        from torchstream.datasets.vidarr import VideoArray,_to_vidarr
        import multiprocessing as mp
        p = mp.Pool(32)
        if self.seq:
            self.samples = p.map(_to_imgseq, self.datapoints)
        else:
            self.samples = p.map(_to_vidarr, self.datapoints)
