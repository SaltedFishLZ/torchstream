import tqdm
import torch
import collections
from torchstream.transforms import Compose, Resize, CenterCrop, CenterSegment
from torchstream.datasets.jesterv1 import JesterV1


def test_jesterv1(ext="avi"):
    # training set size: 118562
    dataset_size = 118562
    dataset_path = "~/Datasets/Jester/Jester-v1-{}".format(ext)
    dataset = JesterV1(root=dataset_path, train=True, ext=ext)
    assert dataset.__len__() == dataset_size, ValueError

    # validation set size 14787
    dataset_size = 14787
    dataset_path = "~/Datasets/Jester/Jester-v1-{}".format(ext)
    dataset = JesterV1(root=dataset_path, train=False, ext=ext)
    assert dataset.__len__() == dataset_size, ValueError

    # validation set size 14787
    dataset_size = 14787
    dataset_path = "~/Datasets/Jester/Jester-v1-{}".format(ext)
    dataset = JesterV1(root=dataset_path, train=False, ext=ext)
    assert dataset.__len__() == dataset_size, ValueError

    # dataloader test
    dataset = JesterV1(root=dataset_path,
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
    # test_jesterv1("avi")
    test_jesterv1("jpg")
