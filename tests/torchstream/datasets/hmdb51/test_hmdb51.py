import collections
import torch
import tqdm
from torchstream.transforms import Compose, Resize, CenterCrop, CenterSegment
from torchstream.datasets.hmdb51 import HMDB51


def test_hmdb51():
    # dataset_len = 6766
    dataset_path = "~/Datasets/HMDB51/HMDB51-avi"
    dataset = HMDB51(root=dataset_path,
                     transform=Compose([CenterSegment(32),
                                        Resize(256),
                                        CenterCrop(224)]),
                     train=True)
    print(dataset.__len__())

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
    test_hmdb51()
