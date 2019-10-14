import collections

import tqdm
import torch
from torchstream.transforms import Compose, Resize, CenterCrop, CenterSegment
from torchstream.io.framesampler import CenterSegmentFrameSampler
from torchstream.datasets.kinetics400 import Kinetics400


def test_kinetics_400():
    # mp4 data
    dataset_path = "~/Datasets/Kinetics/Kinetics-400-mp4"
    dataset = Kinetics400(root=dataset_path,
                          transform=Compose([CenterSegment(32),
                                             Resize(256),
                                             CenterCrop(224)]),
                          train=False, ext="mp4")
    print(dataset.__len__())
    # HACK
    dataset.datapoints = dataset.datapoints[12700:]
    print(dataset.__len__())

    """
    # jpg data
    frame_sampler = CenterSegmentFrameSampler(8)
    dataset_path = "~/Datasets/Kinetics/Kinetics-400-jpg"
    dataset = Kinetics400(root=dataset_path,
                          # remove segment
                          transform=Compose([Resize(256),
                                             CenterCrop(224)]),
                          train=False, ext="jpg",
                          frame_sampler=frame_sampler)
    print(dataset.__len__())
    # HACK
    dataset.datapoints = dataset.datapoints[12700:]
    print(dataset.__len__())
    """

    # loading test
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=32,
                                             num_workers=32,
                                             pin_memory=True)

    num_samples_per_class = collections.OrderedDict()
    for vid, cid in tqdm.tqdm(dataloader):
        for _cid in cid.numpy():
            if _cid in num_samples_per_class:
                num_samples_per_class[_cid] += 1
            else:
                num_samples_per_class[_cid] = 1
    print(num_samples_per_class)


if __name__ == "__main__":
    test_kinetics_400()
