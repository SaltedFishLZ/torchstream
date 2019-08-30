import collections
from torchstream.datasets.hmdb51 import HMDB51


def test_hmdb51():
    # dataset_len = 6766
    dataset_path = "~/Datasets/HMDB51/HMDB51-avi"
    dataset = HMDB51(root=dataset_path, train=True)
    print(dataset.__len__())

    num_samples_per_class = collections.OrderedDict()
    for vid, cid in dataset:
        if cid in num_samples_per_class:
            num_samples_per_class[cid] += 1
        else:
            num_samples_per_class[cid] = 1
    print(num_samples_per_class)

if __name__ == "__main__":
    test_hmdb51()
