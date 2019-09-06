import collections
from torchstream.datasets.sthsthv1 import SomethingSomethingV1


def test_sthsthv1():
    # training set size: 86017
    dataset_size = 86017
    dataset_path = "~/Datasets/Sth-sth/Sth-sth-v1-jpg"
    dataset = SomethingSomethingV1(root=dataset_path, train=True)
    assert dataset.__len__() == dataset_size, ValueError

    # validation set size 11522
    dataset_size = 11522
    dataset_path = "~/Datasets/Sth-sth/Sth-sth-v1-jpg"
    dataset = SomethingSomethingV1(root=dataset_path, train=False)
    assert dataset.__len__() == dataset_size, ValueError

    # validation set size 11522
    dataset_size = 11522
    dataset_path = "~/Datasets/Sth-sth/Sth-sth-v1-avi"
    dataset = SomethingSomethingV1(root=dataset_path, train=False, ext="avi")
    assert dataset.__len__() == dataset_size, ValueError

    num_samples_per_class = collections.OrderedDict()
    for vid, cid in dataset:
        if cid in num_samples_per_class:
            num_samples_per_class[cid] += 1
        else:
            num_samples_per_class[cid] = 1
    print(num_samples_per_class)

if __name__ == "__main__":
    test_sthsthv1()
