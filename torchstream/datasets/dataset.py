"""
"""
import os
import sys
import copy
import pickle
import logging
import hashlib
import multiprocessing as mp

import tqdm
import numpy as np
import torch.utils.data as torchdata

from . import __config__
from .metadata import datapoint, collect, __SUPPORTED_MODALITIES__, __SUPPORTED_VIDEOS__, __SUPPORTED_IMAGES__
from .imgseq import ImageSequence, _to_imgseq
from .vidarr import VideoArray,_to_vidarr
from .utils import filesys

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
        pickle.dump(imgseqs, f)    
    
    return imgseqs



def create_segimgseq(x):
    return SegmentedImageSequence(x)
 
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
    imgseqs = []
    p = mp.Pool(32)
    imgseqs = p.map(create_segimgseq, samples)

    ## dump to cache file
    os.makedirs(CACHE_PATH, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(imgseqs, f)    
    
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
    def __init__(self, root, layout, mod, ext, datapoint_filter=None,
                 transform=None, target_transform=None,
                 **kwargs
                ):
        """
        Args:

        """
        assert isinstance(root, str), TypeError
        assert os.path.exists(root), "Root Path Not Exists"
        assert mod in __SUPPORTED_MODALITIES__, NotImplementedError
        assert ext in __SUPPORTED_MODALITIES__[mod], NotImplementedError

        self.root = root
        self.layout = layout
        self.mod = mod
        self.ext = ext
        self.seq = ext in __SUPPORTED_IMAGES__[mod]
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs
        
        ## collect datapoints
        datapoints = collect.collect_datapoints(root=root, layout=self.layout,
                                                datapoint_filter=datapoint_filter,
                                                mod=mod, ext=ext, **kwargs)
        self.datapoints = datapoints

        import time
        st_time = time.time()
        p = mp.Pool(32)
        if self.seq:
            # Cache Mechanism
            md5 = hashlib.md5(root.encode('utf-8')).hexdigest()
            cache_file = "{}.{}.all{}.imgseqs".format(mod, ext, md5)
            cache_file = os.path.join(CACHE_PATH, cache_file)
            if (os.path.exists(cache_file)
                    and os.path.isfile(cache_file)
                    # and filesys.touch_date(cache_file) > filesys.touch_date(FILE_PATH)
                    and filesys.touch_date(cache_file) > filesys.touch_date(root)
                ):
                warn_str = "[video dataset] find valid cache {}".\
                    format(cache_file)
                logger.warning(warn_str)
                with open(cache_file, "rb") as f:
                    allseqs = pickle.load(f)
            else:
                ## re-generate all image sequences
                allpoints = collect.collect_datapoints(root=root, layout=self.layout,
                                                       mod=mod, ext=ext, **kwargs)
                allseqs = p.map(_to_imgseq, allpoints)
                
                ## dump to cache file
                os.makedirs(CACHE_PATH, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(allseqs, f)                

        else:
            self.samples = p.map(_to_vidarr, self.datapoints)
        ed_time = time.time()
        print("generating time", ed_time - st_time)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        """
        # currently, we return a Numpy ndarray although it
        # may consume too much memory.

        sample = self.samples[idx]
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


def test(dataset, use_tqdm=True):

    test_components = {
        "basic" : True,
        "__len__" : True,
        "__getitem__" : True,
        "dataloader" : True
    }
    import importlib


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

        if hasattr(metaset, "AVI_DATA_PATH"):
            kwargs["root"] = metaset.AVI_DATA_PATH
            kwargs["ext"] = "avi"

        if hasattr(metaset, "__ANNOTATIONS__"):
            kwargs["annots"] = metaset.__ANNOTATIONS__

        if hasattr(metaset, "JPG_FILE_TMPL"):
            kwargs["tmpl"] = metaset.JPG_FILE_TMPL

        if hasattr(metaset, "JPG_IDX_OFFSET"):
            kwargs["offset"] = metaset.JPG_IDX_OFFSET

        testset = VideoDataset(datapoint_filter=metaset.TestsetFilter(),
                               **kwargs)

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

            if test_components["dataloader"]:
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
