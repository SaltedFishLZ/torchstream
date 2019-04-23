import importlib
from vdataset.dataset import VideoDataset

if __name__ == "__main__":

    DATASET = "UCF101"
    dataset_mod = importlib.import_module("vdataset.{}".format(DATASET))

    allset = VideoDataset(
        dataset_mod.prc_data_path, DATASET)
    print(allset.__len__())

    trainset = VideoDataset(
        dataset_mod.prc_data_path, DATASET, part="train")
    print(trainset.__len__())


