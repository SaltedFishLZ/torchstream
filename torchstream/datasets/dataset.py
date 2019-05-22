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

        self.iohandles = []
        for _sample in self.samples:
            if _sample.seq:
                self.iohandles.append(ImageSequence(_sample, **self.kwargs))
            else:
                self.iohandles.append(VideoArray(_sample, **self.kwargs))


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
        # return (a [T][H][W][C] ndarray, class id)
        # ndarray may need to be converted to [T][C][H][W] format in PyTorch
        return (_blob, _cid)


def test():

    test_components = {
        "basic" : True,
        "__len__" : True,
        "__getitem__" : True,
        "torchloader" : False
    }
    
    test_configuration = {
        "datasets"   : ["sth_sth_v1", ]
    }

    for dataset in (test_configuration["datasets"]):
        print("Dataset - [{}]".format(dataset))
        if (test_components["basic"]):

            metaset = importlib.import_module(
                "datasets.metadata.metasets.{}".format(dataset))
            
            allset = VideoDataset(root=metaset.JPG_DATA_PATH,
                                  layout=metaset.__layout__,
                                  lbls=metaset.__LABELS__,
                                  annots=metaset.__ANNOTATIONS__,
                                  tmpl="{0:05d}",
                                  offset=1,
                                  mod="RGB",
                                  ext="jpg"
                                 )
            

            if (test_components["__len__"]):
                print("All samples number:")
                print(allset.__len__())                                        

                if (test_components["__getitem__"]):
                    # train_loader = torchdata.DataLoader(
                    #         allset, batch_size=1, shuffle=True, 
                    #         num_workers=1, pin_memory=True,
                    #         drop_last=True)  # prevent something not % n_GPU
                    # for _i, (inputs, targets) in enumerate(train_loader):
                    #     print(_i, inputs, targets)
                    for _i in tqdm.tqdm(range(allset.__len__())):
                        allset.__getitem__(_i)




if __name__ == "__main__":

    test()
