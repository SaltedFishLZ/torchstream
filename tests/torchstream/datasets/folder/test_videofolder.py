import os
import time
import tqdm
from torchstream.datasets.folder import VideoFolder


def test_videofolder():
    # Currently, we use [HMDB51] as the testing case.
    # You can create symbol link for testing.
    # If you don't have [HMDB51] in your testing machine, this test will
    # give a warning without an assertion.
    dataset_len = 6766
    dataset_path = "~/Datasets/HMDB51/HMDB51-avi"
    # dataset_path = "~/Datasets/Kinetics/Kinetics-400-mp4/val"

    if not os.path.exists(dataset_path):
        print("Warning: dataset missing")
        print(dataset_path)
        return

    dataset = VideoFolder(root=dataset_path)
    assert len(dataset) == dataset_len

    st_time = time.time()

    corrupt_ids = []
    for i, (vid, cid) in enumerate(tqdm.tqdm(dataset)):
        if vid is None:
            corrupt_ids.append(i)
    print("Corrupt Video IDs:")
    print(corrupt_ids)

    ed_time = time.time()
    print("Total Time (Single Process): {}".format(ed_time - st_time))


if __name__ == "__main__":
    test_videofolder()
