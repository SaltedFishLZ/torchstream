# from .vision import VisionDataset
from .folder import VideoFolder

class HMDB51(VideoFolder):
    """HMDB51 Dataset
    """
    def __init__(self, root="~/Datasets/HMDB51/HMDB51-avi", train=True,
                 transform=None, target_transform=None):
        super(HMDB51, self).__init__(root=root,
                                     transform=transform,
                                     target_transform=target_transform)

def test():
    dataset = HMDB51(train=True)
    print(dataset.__len__())

if __name__ == "__main__":
    test()
