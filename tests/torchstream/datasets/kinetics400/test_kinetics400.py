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

    if (ext in SUPPORTED_IMAGES) and test_frame_sampler:
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

    if test_loading:
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=64,
                                                 num_workers=NUM_CPU,
                                                 pin_memory=True)
        num_samples_per_class = collections.OrderedDict()
        for _, cid in tqdm.tqdm(dataloader):
            for _cid in cid.numpy():
                if _cid in num_samples_per_class:
                    num_samples_per_class[_cid] += 1
                else:
                    num_samples_per_class[_cid] = 1
        print(num_samples_per_class)




if __name__ == "__main__":
    test_kinetics400(ext="jpg", test_frame_sampler=True)
    test_kinetics400(ext="jpg")
