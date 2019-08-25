import tqdm
from torchstream.datasets.folder import VideoFolder


def test_videofolder():
    # dataset = VideoFolder(root="~/Datasets/HMDB51/HMDB51-avi")
    dataset = VideoFolder(root="~/Datasets/Kinetics/Kinetics-400-mp4/val")
    print(len(dataset))
    corrupt_ids = []
    for i, (vid, cid) in enumerate(tqdm.tqdm(dataset)):
        if vid is not None:
            shape = vid.shape
        else:
            corrupt_ids.append(i)
    print("Corrupt")
    print(corrupt_ids)

if __name__ == "__main__":
    test_videofolder()
