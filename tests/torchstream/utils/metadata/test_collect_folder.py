import os
import time
from torchstream.utils.metadata import collect_folder


def test_collect_flat():
    # Currently, we use [HMDB51] as the testing case.
    # You can create symbol link for testing.
    # If you don't have [HMDB51] in your testing machine,
    # this test will give a warning and return without an assertion.
    dataset_len = 6766
    dataset_path = "~/Datasets/HMDB51/HMDB51-avi"
    dataset_ext = "avi"

    dataset_path = os.path.expanduser(dataset_path)

    if not os.path.exists(dataset_path):
        print("Warning: dataset missing")
        print(dataset_path)
        return

    st_time = time.time()

    datapoints = collect_folder(dataset_path, dataset_ext)
    print("# datapoints", len(datapoints))
    assert dataset_len == len(datapoints), ValueError

    ed_time = time.time()
    print("Total Time (Single Process): {}".format(ed_time - st_time))


if __name__ == "__main__":
    test_collect_flat()
