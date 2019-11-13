import os
import pickle
import random
import collections

from .vision import VisionDataset
import torchstream.io.backends.opencv as backend
from torchstream.io.datapoint import DataPoint
from torchstream.io.framesampler import FrameSampler
from torchstream.io.__support__ import SUPPORTED_IMAGES, SUPPORTED_VIDEOS
from torchstream.utils.download import download

CACHE_DIR = os.path.expanduser("~/.cache/torchstream/datasets/kinetics400/")
DOWNLOAD_SERVER_PREFIX = (
    "a18:/home/eecs/zhen/video-acc/download/"
    "torchstream/datasets/kinetics400/"
)


class Kinetics400(VisionDataset):
    """
    Args:
        root (str)
        train (bool)
        ext (str)
        frame_sampler (): only valid for image sequences, 
        transform
        target_transform
    """
    def __init__(self, root, train, ext="mp4",
                 frame_sampler=None,
                 transform=None, target_transform=None):
        root = os.path.expanduser(root)

        super(Kinetics400, self).__init__(root=root,
                                          transform=transform,
                                          target_transform=target_transform)
        self.train = train

        self.frame_sampler = None
        if frame_sampler is not None:
            assert ext in SUPPORTED_IMAGES["RGB"], \
                ValueError("frame_sampler is valid for image sequence only!")
            self.frame_sampler = frame_sampler
            assert isinstance(frame_sampler, FrameSampler), TypeError

        # -------------------- #
        #   load datapoints    #
        # -------------------- #

        # assemble paths
        if not (os.path.exists(CACHE_DIR) and
                os.path.isdir(CACHE_DIR)):
            os.makedirs(CACHE_DIR, exist_ok=True)
        if train:
            datapoint_file_name = "kinetics400_{}_training.pkl".format(ext)
        else:
            datapoint_file_name = "kinetics400_{}_validation.pkl".format(ext)
        datapoint_file_path = os.path.join(CACHE_DIR, datapoint_file_name)
        # download when missing
        if not os.path.exists(datapoint_file_path):
            print("downloading Kinetics400 datapoints...")
            download(src=os.path.join(DOWNLOAD_SERVER_PREFIX,
                                      datapoint_file_name),
                     dst=datapoint_file_path)
        # real load
        with open(datapoint_file_path, "rb") as fin:
            self.datapoints = pickle.load(fin)
            assert isinstance(self.datapoints, list), TypeError
            assert isinstance(self.datapoints[0], DataPoint), TypeError
        # replace dataset root
        for dp in self.datapoints:
            dp.root = root
            dp._path = dp.path

        # ------------------ #
        #  load class_to_idx #
        # ------------------ #
        # download labels
        label_file = "kinetics400_labels.txt"
        label_path = os.path.join(CACHE_DIR, label_file)
        if not os.path.exists(label_path):
            print("downloading Kinetics400 label...")
            label_src = os.path.join(DOWNLOAD_SERVER_PREFIX, label_file)
            download(label_src, label_path)
        # build class label to class id mapping (a dictionary)
        self.class_to_idx = collections.OrderedDict()
        with open(label_path, "r") as fin:
            for _line in fin:
                text = _line.split('\n')[0]
                text = text.split(' ')
                self.class_to_idx[' '.join(text[1:])] = int(text[0]) - 1

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        datapoint = self.datapoints[index]

        if datapoint.ext in SUPPORTED_VIDEOS["RGB"]:
            loader = backend.video2ndarray
            path = datapoint.path
            varray = loader(path)
        elif datapoint.ext in SUPPORTED_IMAGES["RGB"]:
            loader = backend.frames2ndarray
            fpaths = datapoint.framepaths
            if self.frame_sampler is not None:
                fpaths = self.frame_sampler(fpaths)
            varray = loader(fpaths)

        label = datapoint.label
        target = self.class_to_idx[label]

        if self.transform is not None:
            varray = self.transform(varray)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return varray, target

    def gen_index(self, num):
        """generate indices
        Args:
            num (int), number of datapoints
        """
        assert self.train, ValueError("training set only")
        total = len(self.datapoints)
        index = random.sample(range(total), num)
        index.sort()
        return index

    def holdout(self, index, remove=True):
        """
        Args:
            index (list), index datapoints needs to be removed/perserved
                only valid for training set
        """
        assert self.train, ValueError("training set only")
        index = set(index)
        new_points = []
        for _idx, _dp in enumerate(self.datapoints):
            hit = _idx in index
            if hit ^ remove:
                new_points.append(_dp)
        self.datapoints = new_points
