import collections
import torch
import tqdm
from torchstream.transforms import Compose, Resize, CenterCrop, CenterSegment
from torchstream.datasets.sthsthv1 import SomethingSomethingV1


def test_sthsthv1(ext="avi"):
    # training set size: 86017
    dataset_size = 86017
    dataset_path = "~/Datasets/Sth-sth/Sth-sth-v1-{}".format(ext)
    dataset = SomethingSomethingV1(root=dataset_path, train=True)
    assert dataset.__len__() == dataset_size, ValueError

    # validation set size 11522
    dataset_size = 11522
    dataset_path = "~/Datasets/Sth-sth/Sth-sth-v1-{}".format(ext)
    dataset = SomethingSomethingV1(root=dataset_path, train=False)
    assert dataset.__len__() == dataset_size, ValueError

    # validation set size 11522
    dataset_size = 11522
    dataset_path = "~/Datasets/Sth-sth/Sth-sth-v1-{}".format(ext)
    dataset = SomethingSomethingV1(root=dataset_path,
                                   transform=Compose([CenterSegment(32),
                                                      Resize(256),
                                                      CenterCrop(224)]),
                                   train=False, ext=ext)
    assert dataset.__len__() == dataset_size, ValueError

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
    test_sthsthv1(ext="avi")
    test_sthsthv1(ext="jpg")
