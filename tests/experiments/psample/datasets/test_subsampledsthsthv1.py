import os
import sys
import math
import collections


FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)
PRJ_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.dirname(DIR_PATH)))
    )

sys.path.append("experiments/psample")

from datasets import SubsampledSomethingSomethingV1  # noqa: E402


def test_subsampledsthsthv1():
    # original training set size: 86017
    original_dataset_size = 86017
    sample_duration = 10
    dataset_size = math.ceil(original_dataset_size / float(sample_duration))
    dataset_path = "~/Datasets/Sth-sth/Sth-sth-v1-jpg"
    dataset = SubsampledSomethingSomethingV1(root=dataset_path, train=True,
                                             sample_duration=sample_duration)
    assert dataset.__len__() == dataset_size, ValueError
    # inverse sampling
    dataset = SubsampledSomethingSomethingV1(root=dataset_path, train=True,
                                             sample_duration=sample_duration,
                                             inverse=True)
    dataset_size = original_dataset_size - dataset_size
    assert dataset.__len__() == dataset_size, ValueError

    # original validation set size 11522
    original_dataset_size = 11522
    sample_duration = 9
    dataset_size = math.ceil(original_dataset_size / float(sample_duration))
    dataset_path = "~/Datasets/Sth-sth/Sth-sth-v1-jpg"
    dataset = SubsampledSomethingSomethingV1(root=dataset_path, train=False,
                                             sample_duration=sample_duration)
    assert dataset.__len__() == dataset_size, ValueError
    # inverse sampling
    dataset = SubsampledSomethingSomethingV1(root=dataset_path, train=False,
                                             sample_duration=sample_duration,
                                             inverse=True)
    dataset_size = original_dataset_size - dataset_size
    assert dataset.__len__() == dataset_size, ValueError
    # iterating over the dataset
    num_samples_per_class = collections.OrderedDict()
    for vid, cid in dataset:
        if cid in num_samples_per_class:
            num_samples_per_class[cid] += 1
        else:
            num_samples_per_class[cid] = 1
    print(num_samples_per_class)


if __name__ == "__main__":
    test_subsampledsthsthv1()
