"""
"""
import os
import sys
import copy
import pickle
import logging
import importlib
import multiprocessing as mp

import tqdm
import numpy as np
import torch.utils.data as torchdata

from . import __config__
from .metadata import sample, collect
from .imgseq import ImageSequence, ClippedImageSequence, SegmentedImageSequence
from .vidarr import VideoArray
from .utils.cache import hashid, hashstr

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)
CACHE_PATH = os.path.join(DIR_PATH, ".cache")

# ---------------------------------------------------------------- #
#                  Configuring Python Logger                       #
# ---------------------------------------------------------------- #

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
if __config__.__VERY_VERY_VERBOSE__:
    logger.setLevel(logging.INFO)
elif __config__.__VERY_VERBOSE__:
    logger.setLevel(logging.WARNING)
elif __config__.__VERBOSE__:
    logger.setLevel(logging.ERROR)
else:
    logger.setLevel(logging.CRITICAL)



# ------------------------------------------------------------------------- #
#                          Internal Functions                               #
# ------------------------------------------------------------------------- #

def generate_imgseqs(samples, **kwargs):
    # TODO: hash kwargs
    cache_file = "imgseqs{}.pkl".format(hashid(samples))
    cache_file = os.path.join(CACHE_PATH, cache_file)

    if (
        os.path.exists(cache_file)
        and os.path.isfile(cache_file)
        # and touch_date(cache_file) > touch_date(FILE_PATH)
    ):              
        ## find valid cache
        warn_str = "[generate_imgseqs] find valid cache {}".\
            format(cache_file)
        logger.warning(warn_str)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Generating Image Sequences...")
    def create_imgseq(x):
        return ImageSequence(x)
    imgseqs = []
    p = mp.Pool(32)
    imgseqs = p.map(create_imgseq, samples)


    ## dump to cache file
    os.makedirs(CACHE_PATH, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(samples, f)    
    
    return imgseqs


def generate_clipimgseqs(samples, **kwargs):
    # TODO: hash kwargs
    cache_file = "clipimgseqs{}.pkl".format(hashid(samples))
    cache_file = os.path.join(CACHE_PATH, cache_file)

    if (
        os.path.exists(cache_file)
        and os.path.isfile(cache_file)
        # and touch_date(cache_file) > touch_date(FILE_PATH)
    ):              
        ## find valid cache
        warn_str = "[generate_clipimgseqs] find valid cache {}".\
            format(cache_file)
        logger.warning(warn_str)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Generating Clipped Image Sequences...")
    def create_imgseq(x):
        return ClippedImageSequence(x)
    imgseqs = []
    p = mp.Pool(32)
    imgseqs = p.map(create_imgseq, samples)


    ## dump to cache file
    os.makedirs(CACHE_PATH, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(samples, f)    
    
    return imgseqs


def generate_segimgseqs(samples, **kwargs):
    # TODO: hash kwargs
    print(type(samples))
    cache_file = "segimgseqs{}.pkl".format(hashid(samples))
    cache_file = os.path.join(CACHE_PATH, cache_file)

    if (
        os.path.exists(cache_file)
        and os.path.isfile(cache_file)
        # and touch_date(cache_file) > touch_date(FILE_PATH)
    ):              
        ## find valid cache
        warn_str = "[generate_segimgseqs] find valid cache {}".\
            format(cache_file)
        logger.warning(warn_str)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Generating Segmented Image Sequences...")

    def create_imgseq(x):
        return SegmentedImageSequence(x)
    imgseqs = []
    p = mp.Pool(32)
    imgseqs = p.map(create_imgseq, samples)

    ## dump to cache file
    os.makedirs(CACHE_PATH, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(samples, f)    
    
    return imgseqs



# ------------------------------------------------------------------------- #
#                   Main Classes (To Be Used outside)                       #
# ------------------------------------------------------------------------- #

## dataset class for video recognition
#  More details.
class VideoDataset(torchdata.Dataset):
    """dataset class for video recognition
    Args
        optional
        filter: Sample filter

    """
    def __init__(self, root, layout, lbls, mod, ext,
                 transform=None, target_transform=None,
                 **kwargs
                ):
        """
        Args:

        """
        self.root = root
        self.layout = layout
        self.lbls = lbls
        self.mod = mod
        self.ext = ext
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs

        
        ## collect samples
        _sampleset = collect.collect_samples(root=root,
                                layout=self.layout, lbls=self.lbls,
                                mod=mod, ext=ext,
                                **kwargs
                               )

        logger.critical("Turning set to list...")
        _samplelist = list(_sampleset)
        logger.critical("Sorting list...")
        _samplelist.sort()
        self.samples = _samplelist

        self.iohandles = generate_imgseqs(self.samples, **kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        """
        # currently, we intended to return a Numpy ndarray although it
        # may consume too much memory.
        _iohanlde = self.iohandles[idx]
        _blob = np.array(_iohanlde)
        _sample = self.samples[idx]
        _cid = self.lbls[_sample.lbl]
        if self.transform is not None:
            _blob = self.transform(_blob)
        if self.target_transform is not None:
            _cid = self.target_transform(_cid)
        # return (a [T][H][W][C] ndarray, class id)
        # ndarray may need to be converted to [T][C][H][W] format in PyTorch
        return (_blob, _cid)


class ClippedVideoDataset(VideoDataset):
    """
    """
    def __init__(self, root, layout, lbls, mod, ext,
                 transform=None, target_transform=None,
                 **kwargs
                ):
        assert ext == "jpg", TypeError
        self.root = root
        self.layout = layout
        self.lbls = lbls
        self.mod = mod
        self.ext = ext
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs

        
        ## collect samples
        _sampleset = collect.collect_samples(root=root,
                                layout=self.layout, lbls=self.lbls,
                                mod=mod, ext=ext,
                                **kwargs
                               )

        logger.critical("Turning set to list...")
        _samplelist = list(_sampleset)
        logger.critical("Sorting list...")
        _samplelist.sort()
        self.samples = _samplelist

        self.iohandles = generate_clipimgseqs(self.samples, **kwargs)


class SegmentedVideoDataset(VideoDataset):
    """
    """
    def __init__(self, root, layout, lbls, mod, ext,
                 transform=None, target_transform=None,
                 **kwargs
                ):
        assert ext == "jpg", TypeError
        self.root = root
        self.layout = layout
        self.lbls = lbls
        self.mod = mod
        self.ext = ext
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs

        
        ## collect samples
        _sampleset = collect.collect_samples(root=root,
                                layout=self.layout, lbls=self.lbls,
                                mod=mod, ext=ext,
                                **kwargs
                               )

        logger.critical("Turning set to list...")
        _samplelist = list(_sampleset)
        logger.critical("Sorting list...")
        _samplelist.sort()
        self.samples = _samplelist

        self.iohandles = generate_segimgseqs(self.samples, **kwargs)




def test(dataset, use_tqdm=True):

    test_components = {
        "basic" : True,
        "__len__" : True,
        "__getitem__" : True,
        "torchloader" : True
    }
    

    print("Dataset - [{}]".format(dataset))
    if (test_components["basic"]):
        metaset = importlib.import_module(
            "torchstream.datasets.metadata.metasets.{}".format(dataset))

        kwargs = {
            "root": metaset.JPG_DATA_PATH,
            "layout": metaset.__layout__,
            "lbls": metaset.__LABELS__,
            "mod": "RGB",
            "ext": "jpg",
        }

        # if hasattr(metaset, "AVI_DATA_PATH"):
        #     kwargs["root"] = metaset.AVI_DATA_PATH
        #     kwargs["ext"] = "avi"

        if hasattr(metaset, "__ANNOTATIONS__"):
            kwargs["annots"] = metaset.__ANNOTATIONS__

        if hasattr(metaset, "JPG_FILE_TMPL"):
            kwargs["tmpl"] = metaset.JPG_FILE_TMPL

        if hasattr(metaset, "JPG_IDX_OFFSET"):
            kwargs["offset"] = metaset.JPG_IDX_OFFSET

        # allset = VideoDataset(root=metaset.AVI_DATA_PATH,
        #                       layout=metaset.__layout__,
        #                       lbls=metaset.__LABELS__,
        #                       mod="RGB",
        #                       ext="avi"
        #                      )
        testset = SegmentedVideoDataset(
                              filter=metaset.TestsetFilter(),
                              seg_num=7,
                              **kwargs
                             )

        if test_components["__len__"]:
            # print("All samples number:")
            # print(allset.__len__())
            print("Testing samples number:")
            print(testset.__len__())

            if test_components["__getitem__"]:

                irange = range(testset.__len__())
                if use_tqdm:
                    irange = tqdm.tqdm(irange)
                for _i in irange:
                    testset.__getitem__(_i)

            if test_components["torchloader"]:
                print("Testing torch dataloader")
                train_loader = torchdata.DataLoader(
                        testset, batch_size=1, shuffle=True,
                        num_workers=8, pin_memory=True,
                        drop_last=True)  # prevent something not % n_GPU
                for _i, (inputs, _) in enumerate(train_loader):
                    print(inputs.shape)
                    

if __name__ == "__main__":
    print(sys.argv)
    for _i in range(1, len(sys.argv)):
        test(sys.argv[_i])
