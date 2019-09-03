import os
import pickle
import collections

from .vision import VisionDataset
from torchstream.io.datapoint import DataPoint
import torchstream.io.backends.opencv as backend
from torchstream.io.__support__ import SUPPORTED_IMAGES, SUPPORTED_VIDEOS
from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download")
DOWNLOAD_SRC_DIR = "torchstream/datasets/ucf101/"

CACHE_DIR = os.path.expanduser("~/.cache/torchstream/datasets/ucf101/")


class UCF101(VisionDataset):

    def __init__(self, root, train, split=1, class_to_idx=None,
                 ext="avi",
                 transform=None, target_transform=None):
        root = os.path.expanduser(root)

        super(UCF101, self).__init__(root=root,
                                     transform=transform,
                                     target_transform=target_transform)
        # -------------------- #
        #   load datapoints    #
        # -------------------- #

        # assemble paths
        if not (os.path.exists(CACHE_DIR) and
                os.path.isdir(CACHE_DIR)):
            os.makedirs(CACHE_DIR, exist_ok=True)
        if train:
            datapoint_filename = "ucf101_training_split{}.pkl".format(split)
        else:
            datapoint_filename = "ucf101_testing_split{}.pkl".format(split)
        datapoint_filepath = os.path.join(CACHE_DIR, datapoint_filename)
        # download when missing
        if not os.path.exists(datapoint_filepath):
            print("downloading UCF101 datapoints...")
            download(src=os.path.join(DOWNLOAD_SERVER_PREFIX,
                                      DOWNLOAD_SRC_DIR,
                                      datapoint_filename),
                     dst=datapoint_filepath,
                     backend="rsync")
        # real load
        with open(datapoint_filepath, "rb") as fin:
            self.datapoints = pickle.load(fin)
            assert isinstance(self.datapoints, list), TypeError
            assert isinstance(self.datapoints[0], DataPoint), TypeError
        # replace dataset root
        for dp in self.datapoints:
            dp.root = root
            dp.ext = ext
            dp._path = dp.path

        # ------------------ #
        #  load class_to_idx #
        # ------------------ #
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            class_to_idx_filename = "ucf101_class_to_idx.pkl"
            class_to_idx_filepath = os.path.join(CACHE_DIR,
                                                 class_to_idx_filename)
            # download when missing
            if not os.path.exists(class_to_idx_filepath):
                print("downloading UCF101 class_to_idx...")
                download(src=os.path.join(DOWNLOAD_SERVER_PREFIX,
                                          DOWNLOAD_SRC_DIR,
                                          class_to_idx_filename),
                         dst=class_to_idx_filepath,
                         backend="rsync")
            # load class_to_idx
            with open(class_to_idx_filepath, "rb") as fin:
                self.class_to_idx = pickle.load(fin)
        # sanity check
        # print(self.class_to_idx)
        assert isinstance(self.class_to_idx, dict), TypeError

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        datapoint = self.datapoints[index]

        if datapoint.ext in SUPPORTED_VIDEOS["RGB"]:
            loader = backend.video2ndarray
        elif datapoint.ext in SUPPORTED_IMAGES["RGB"]:
            loader = backend.frames2ndarray

        path = datapoint._path
        varray = loader(path)

        label = datapoint.label
        target = self.class_to_idx[label]

        if self.transform is not None:
            varray = self.transform(varray)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return varray, target
