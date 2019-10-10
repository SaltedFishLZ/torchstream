import collections

import tqdm
import torch
from torchstream.transforms import Compose, Resize, CenterCrop, CenterSegment
from torchstream.io.framesampler import CenterSegmentFrameSampler
from torchstream.datasets.hmdb51 import HMDB51


def test_hmdb51():
    # split 1, training set
    # dataset_len = 6766
    dataset_path = "~/Datasets/HMDB51/HMDB51-avi"
    dataset = HMDB51(root=dataset_path,
                     transform=Compose([CenterSegment(32),
                                        Resize(256),
                                        CenterCrop(224)]),
                     train=True)
    # assert dataset.__len__() == 6766, ValueError

    dataset_path = "~/Datasets/HMDB51/HMDB51-jpg"
    dataset = HMDB51(root=dataset_path,
                     transform=Compose([CenterSegment(32),
                                        Resize(256),
                                        CenterCrop(224)]),
                     train=True)

    frame_sampler = CenterSegmentFrameSampler(8)
    dataset_path = "~/Datasets/HMDB51/HMDB51-jpg"
    dataset = HMDB51(root=dataset_path,
                     transform=Compose([CenterSegment(32),
                                        Resize(256),
                                        CenterCrop(224)]),
                     train=True, ext="jpg",
                     frame_sampler=frame_sampler)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=16,
                                             num_workers=4,
                                             pin_memory=True)

    num_samples_per_class = collections.OrderedDict()
    for vid, cid in tqdm.tqdm(dataloader):
        for _cid in cid.numpy():
            if _cid in num_samples_per_class:
                num_samples_per_class[_cid] += 1
            else:
                num_samples_per_class[_cid] = 1
    print(num_samples_per_class)

    return

    # split 1, validation set
    # dataset_len = 3750
    dataset_path = "~/Datasets/HMDB51/HMDB51-avi"
    dataset = HMDB51(root=dataset_path,
                     transform=Compose([CenterSegment(32),
                                        Resize(256),
                                        CenterCrop(224)]),
                     train=False)
    assert dataset.__len__() == 3750, ValueError


if __name__ == "__main__":
    test_hmdb51()
