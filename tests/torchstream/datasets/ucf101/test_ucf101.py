import collections
import torch
import tqdm
from torchstream.transforms import Compose, Resize, CenterCrop, CenterSegment
from torchstream.datasets.ucf101 import UCF101


def test_ucf101():
    dataset_path = "~/Datasets/UCF101/UCF101-avi"
    dataset = UCF101(root=dataset_path,
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
    test_ucf101()
