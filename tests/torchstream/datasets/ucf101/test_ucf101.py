import collections
from torchstream.datasets.ucf101 import UCF101


def test_ucf101():
    # dataset_len = 6766
    dataset_path = "~/Datasets/UCF101/UCF101-avi"
    dataset = UCF101(root=dataset_path, train=True)
    print(dataset.__len__())

    num_samples_per_class = collections.OrderedDict()
    for vid, cid in dataset:
        if cid in num_samples_per_class:
            num_samples_per_class[cid] += 1
        else:
            num_samples_per_class[cid] = 1
    print(num_samples_per_class)

if __name__ == "__main__":
    test_ucf101()
