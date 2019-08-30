from torchstream.datasets.hmdb51 import HMDB51


def test_hmdb51():
    # dataset_len = 6766
    dataset_path = "~/Datasets/HMDB51/HMDB51-avi"
    dataset = HMDB51(root=dataset_path, train=True)
    print(dataset.__len__())

    for vid, cid in dataset:
        print(cid)


if __name__ == "__main__":
    test_hmdb51()
