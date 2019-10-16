import collections
import torch
import tqdm
from torchstream.transforms import Compose, Resize, CenterCrop, CenterSegment
from torchstream.datasets.ucf101 import UCF101


def test_ucf101(ext="avi", split=1, train=False, test_loading=True):

    dataset_path = "~/Datasets/UCF101/UCF101-{}".format(ext)

    dataset = UCF101(root=dataset_path, ext=ext, split=split,
                     transform=Compose([CenterSegment(32),
                                        Resize(256),
                                        CenterCrop(224)]),
                     train=train)
    print("{} set length".format("training" if train else "validation"))
    print(dataset.__len__())

    if test_loading:
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
    print("*" * 80)
    print("AVI, Split 1, Val")
    print("*" * 80)
    test_ucf101(ext="avi", split=1)
    test_ucf101(ext="avi", split=2)
    test_ucf101(ext="avi", split=3)

    test_ucf101(ext="jpg", split=1)
    test_ucf101(ext="jpg", split=2)
    test_ucf101(ext="jpg", split=3)
