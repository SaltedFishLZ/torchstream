import collections
from torchstream.datasets.kinetics400 import Kinetics400


def test_kinetics_400():
    dataset_path = "/dnn/data/Kinetics/Kinetics-400-mp4"
    dataset = Kinetics400(root=dataset_path, train=False)
    print(dataset.__len__())

    num_samples_per_class = collections.OrderedDict()
    for vid, cid in dataset:
        if cid in num_samples_per_class:
            num_samples_per_class[cid] += 1
        else:
            num_samples_per_class[cid] = 1

if __name__ == "__main__":
    test_kinetics_400()
