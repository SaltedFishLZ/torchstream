import collections

import tqdm
import torch
from torchstream.transforms import Compose, Resize, CenterCrop, CenterSegment
from torchstream.io.framesampler import CenterSegmentFrameSampler
from torchstream.datasets.kinetics400 import Kinetics400


def test_kinetics400(ext="jpg", train=False,
                      test_loading=True,
                      test_frame_sampler=False):

    dataset_path = "~/Datasets/Kinetics/Kinetics-400-{}".format(ext)

    if (ext in SUPPORTED_IMAGES) and test_frame_sampler:
        frame_sampler = CenterSegmentFrameSampler(8)
        dataset = HMDB51(root=dataset_path,
                         transform=Compose([CenterSegment(32),
                                            Resize(256),
                                            CenterCrop(224)]),
                         ext=ext, split=split, train=train,
                         frame_sampler=frame_sampler)
    else:
        dataset = HMDB51(root=dataset_path,
                         transform=Compose([CenterSegment(32),
                                            Resize(256),
                                            CenterCrop(224)]),
                          ext=ext, split=split, train=train)

    print("{} set length".format("training" if train else "validation"))
    print(dataset.__len__())

    if test_loading:
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=16,
                                                 num_workers=4,
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
