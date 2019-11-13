import collections
import multiprocessing

import tqdm
import torch
from torchstream.transforms import Compose, Resize, CenterCrop, CenterSegment
from torchstream.io.framesampler import CenterSegmentFrameSampler
from torchstream.io import SUPPORTED_IMAGES
from torchstream.datasets.kinetics400 import Kinetics400

NUM_CPU = multiprocessing.cpu_count()


def test_kinetics400(ext="jpg", train=False,
                     test_loading=True,
                     test_frame_sampler=False):

    dataset_path = "~/Datasets/Kinetics/Kinetics-400-{}".format(ext)

    if (ext in SUPPORTED_IMAGES["RGB"]) and test_frame_sampler:
        frame_sampler = CenterSegmentFrameSampler(8)
        dataset = Kinetics400(root=dataset_path,
                              transform=Compose([Resize(256),
                                                 CenterCrop(224)]),
                              ext=ext, train=train,
                              frame_sampler=frame_sampler)
    else:
        dataset = Kinetics400(root=dataset_path,
                              transform=Compose([CenterSegment(8),
                                                 Resize(256),
                                                 CenterCrop(224)]),
                              ext=ext, train=train)

    print("{} set length".format("training" if train else "validation"))
    print(dataset.__len__())

    holdout_index = dataset.gen_index(10000)
    print(holdout_index)
    torch.save(holdout_index, "kinetics400-{}.holdout".format(ext))

    dataset.holdout(holdout_index, remove=True)
    print(dataset.__len__())


if __name__ == "__main__":
    print("*" * 80)
    print("JPG, Split 1, Val, FrameSampler(8)")
    print("*" * 80)
    test_kinetics400(ext="jpg", test_frame_sampler=True)

    print("*" * 80)
    print("JPG, Split 1, Train, FrameSampler(8)")
    print("*" * 80)
    test_kinetics400(ext="jpg", test_frame_sampler=True, train=True)
