"""
"""
import os
import sys
import copy
import logging
import importlib

import tqdm
import numpy as np
import torch.utils.data as torchdata

from . import __config__
from .metadata import sample, collect
from .imgseq import ImageSequence, ClippedImageSequence, SegmentedImageSequence
from .vidarr import VideoArray

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
#                   Main Classes (To Be Used outside)                       #
# ------------------------------------------------------------------------- #

## dataset class for video recognition
#  More details.
class VideoDataset(torchdata.Dataset):
    """dataset class for video recognition
    """

    def __init__(self, root, layout, lbls, 
                 mod, ext,
                 sample_filter=None, 
                 **kwargs
                ):
        """
        Args:
            configs: a list of dict 
            {
                "root": <root path>,
                "mod": <modality>,
                "ext": <file extension>
            }
        """
        self.root = root        
        self.layout = layout
        self.lbls = lbls
        self.mod = mod
        self.ext = ext
        self.sample_filter = sample_filter
        self.kwargs = kwargs

        
        ## collect samples
        _sampleset = collect.collect_samples(root=root,
                                layout=self.layout, lbls=self.lbls,
                                mod=mod, ext=ext, **kwargs
                               )
        _samplelist = list(_sampleset)
        _samplelist.sort()
        self.samples = _samplelist



    def __len__(self):
        return len(self.samplelists[0])

    def __getitem__(self, idx):
        """
        """
        # currently, we intended to return a Numpy ndarray although it
        # may consume too much memory.

        _blobs = []
        for _i in len(self.configs):
            _blob = np.array(self.iohandlelists[_i][idx])
            _blobs.append(_blob)

        _sample = self.samplelists[0][idx]
        _cid = self.lbls[_sample.name]

        # return (a [T][H][W][C] ndarray, class id)
        # ndarray may need to be converted to [T][C][H][W] format in PyTorch
        return(_blobs, _cid)


def test():

    test_components = {
        'basic' : True,
        '__len__' : True,
        '__getitem__' : True
    }
    
    test_configuration = {
        'datasets'   : ["weizmann", ]
    }

    for DATASET in (test_configuration['datasets']):
        print("Dataset - [{}]".format(DATASET))
        if (test_components['basic']):

            dset = importlib.import_module(
                "vdataset.metasets.{}".format(DATASET))
            allset = VideoDataset(
                dset.RAW_DATA_PATH, DATASET,
                modalities={'RGB': "avi"},
                split=constant.ALLSET,
                img_file_temp="{0:05d}",
                img_idx_offset=1)
            trainset = VideoDataset(
                dset.RAW_DATA_PATH, DATASET,
                clip_len=4,
                modalities={'RGB': "avi"},
                split=constant.TRAINSET)
            testset = VideoDataset(
                dset.RAW_DATA_PATH, DATASET,
                clip_len=4,
                modalities={'RGB': "avi"},
                split=constant.TESTSET)

            if (test_components['__len__']):
                print("All samples number:")
                print(allset.__len__())
                print("Training Set samples number:")
                print(trainset.__len__())
                print("Testing Set samples number:")
                print(testset.__len__())
            
                # if (test_components['__getitem__']):
                #     # print(allset.__getitem__(allset.__len__()-1))
                #     for _idx in tqdm.tqdm(range(allset.__len__())):
                #         allset.__getitem__(_idx)
                #     for _idx in tqdm.tqdm(range(trainset.__len__())):
                #         trainset.__getitem__(_idx)
                #     for _idx in tqdm.tqdm(range(testset.__len__())):
                #         testset.__getitem__(_idx)                                            

                if (test_components['__getitem__']):
                    train_loader = torch.utils.data.DataLoader(
                            trainset, batch_size=1, shuffle=True, 
                            num_workers=1, pin_memory=True,
                            drop_last=True)  # prevent something not % n_GPU
                    for _i, (inputs, targets) in enumerate(train_loader):
                        print(_i, inputs)



if __name__ == "__main__":

    if __test__:
        test()
