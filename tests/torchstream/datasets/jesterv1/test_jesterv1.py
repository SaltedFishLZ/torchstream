import collections
from torchstream.datasets.jesterv1 import JesterV1


def test_jesterv1():
    # training set size: 118562
    dataset_size = 118562
    dataset_path = "~/Datasets/Jester/Jester-v1-jpg"
    dataset = SomethingSomethingV1(root=dataset_path, train=True)
    assert dataset.__len__() == dataset_size, ValueError

    # validation set size 14787
    dataset_size = 14787
    dataset_path = "~/Datasets/Jester/Jester-v1-jpg"
    dataset = SomethingSomethingV1(root=dataset_path, train=False)
    assert dataset.__len__() == dataset_size, ValueError

    # validation set size 14787
    dataset_size = 14787
    dataset_path = "~/Datasets/Jester/Jester-v1-avi"
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
    test_jesterv1()
