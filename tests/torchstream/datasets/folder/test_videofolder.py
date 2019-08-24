import tqdm
from torchstream.datasets.folder import VideoFolder


def test_videofolder():
    dataset = VideoFolder(root="~/Datasets/HMDB51/HMDB51-avi")
    print(len(dataset))
    for vid, cid in tqdm.tqdm(dataset):
        shape = vid.shape

if __name__ == "__main__":
    test_videofolder()