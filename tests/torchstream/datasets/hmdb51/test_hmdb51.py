from torchstream.datasets.hmdb51 import HMDB51

def test():
    dataset = HMDB51(train=True)
    print(dataset.__len__())

if __name__ == "__main__":
    test()
