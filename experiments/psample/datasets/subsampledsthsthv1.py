from torchstream.datasets import SomethingSomethingV1


class SubsampledSomethingSomethingV1(SomethingSomethingV1):
    """Interleaving Subsampled SomethingSomethingV1
    Args:
        root, train, class_to_idx, ext, fpath_offset, fpath_tmpl:
            the same as SomethingSomethingV1
        sample_duration (int):
            sample duration
        inverse (bool):
            choose those idx % duration != 0
    """
    def __init__(self, root, train, sample_duration, inverse=False,
                 class_to_idx=None, ext="jpg",
                 fpath_offset=1, fpath_tmpl="{:05d}",
                 transform=None, target_transform=None):
        assert isinstance(sample_duration, int), TypeError
        assert sample_duration > 0, ValueError

        self.sample_duration = sample_duration
        self.inverse = inverse
        super(SubsampledSomethingSomethingV1, self).__init__(
            root=root, train=train, class_to_idx=class_to_idx,
            ext=ext, fpath_offset=fpath_offset, fpath_tmpl=fpath_tmpl,
            transform=transform, target_transform=target_transform
        )

        # subsample datapoints
        new_datapoints = []
        for _i, dp in enumerate(self.datapoints):
            if not self.inverse:
                if _i % self.sample_duration == 0:
                    new_datapoints.append(dp)
            else:
                if _i % self.sample_duration != 0:
                    new_datapoints.append(dp)
        self.datapoints = new_datapoints
