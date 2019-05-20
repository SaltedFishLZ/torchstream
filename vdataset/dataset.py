# -*- coding: utf-8 -*-
import os
import sys
import copy
import logging
import cProfile
import importlib

import tqdm
import torch
import torch.utils.data as torchdata


from .constant import \
    __test__, __profile__, __strict__, __verbose__, __vverbose__, \
    __supported_modalities__, __supported_modality_files__, \
    __supported_video_files__, __supported_color_space__, \
    __supported_dataset_styles__, __supported_datasets__

from . import video, metadata, constant

__verbose__ = False

## dataset class for video recognition
#  
#  More details.
class VideoDataset(torchdata.Dataset):
    """
    dataset class for video recognition

    NOTE: TODO: Currently, This shall be an abstract base class.
    It should never be used in deployment !!!
    """
    ## Init
    #  @param root str: root path of the dataset
    #  @param name str: dataset name
    #  @param split int: training/testing/validation split
    #  @param modalities dict: input data format of each modality.
    #  key - modality, value - file extension(s) for certain modality.
    def __init__(self, root, name,
                 split=constant.TRAINSET,
                 modalities={'RGB': constant.IMGSEQ},
                 *args, **kwargs
                ):
        """
        @param root str: root path of the dataset
        @param name str: dataset name
        @param split int: training/testing/validation split
        @param modalities dict: input data format of each modality.
        key - modality, value - file extension(s) for certain modality.
        """
        # TODO: support multiple input data modalities

        # santity check
        assert os.path.exists(root), "Dataset path not exists"
        assert name in __supported_datasets__, "Unsupported Dataset"
        assert (1==len(modalities)), "Only support 1 data modality now"
        for _mod in modalities:
            assert _mod in __supported_modalities__, 'Unsupported Modality'
            _ext = modalities[_mod]
            assert _ext in __supported_modality_files__[_mod],\
                ("Unspported input file type: {} for modality: {}".\
                    format(_ext, _mod))

        self.root = root
        self.name = name
        self.style = __supported_datasets__[self.name]
        self.dsetmod = importlib.import_module("vdataset.metasets.{}".format(self.name))
        self.labels = self.dsetmod.__labels__
        self.split = split
        self.modalities = modalities
        self.metadatas = dict()
        self.kwargs = kwargs

        # filter to select metadata
        if self.split != constant.ALLSET:
            if self.split == constant.TRAINSET:
                sample_filter = self.dsetmod.TrainsetFilter()
            elif self.split == constant.VALSET:
                sample_filter = self.dsetmod.ValsetFilter()
            elif self.split == constant.TESTSET:
                sample_filter = self.dsetmod.TestsetFilter()
            else:
                raise NotImplementedError
        else:
            sample_filter = None

        for mod in self.modalities:
            # collect metadata
            ext = modalities[mod]
            
            collector = metadata.Collector(self.root, self.dsetmod,
                                           mod=mod, ext=ext,
                                           sfilter=sample_filter
                                           )
            sample_set = collector.collect_samples()
            
            # append results
            self.metadatas[mod] = sample_set.get_samples()

    def __len__(self):
        return len(self.metadatas["RGB"])

    def __getitem__(self, idx):
        """
        """
        # TODO: fuse multiple modalities
        # NOTE:
        # currently, we intended to return a Numpy ndarray although it
        # may consume too much memory.

        for _modality in self.modalities:
            ## get sample's metadata of a certain modality
            _sample_metadata = (self.metadatas[_modality])[idx]
            _ext = self.modalities[_modality]
            _path = _sample_metadata.path
            _cid = _sample_metadata.cid

            ## Deal with image sequences
            #  get all frames as a varray
            if _ext == constant.IMGSEQ:
                _ext = constant.IMGEXT
                _seq = video.ImageSequence(_path, ext=_ext, **self.kwargs)
                _blob = _seq.get_varray()
            ## Deal with video files
            #  get varray directly
            else:
                _blob = video.video2ndarray(_path)

        # return (a [T][H][W][C] ndarray, class id)
        # ndarray may need to be converted to [T][C][H][W] format in PyTorch
        return(_blob, _cid)



class ClippedVideoDataset(VideoDataset):
    """

    """
    def __init__(self, root, name, clip_len,
                 split=constant.TRAINSET,
                 modalities={'RGB': constant.IMGSEQ},
                 *args, **kwargs
                ):
        super(ClippedVideoDataset, self).__init__(
            root, name, split, modalities, args, **kwargs
            )
        self.clip_len = clip_len

    def __getitem__(self, idx):
        """
        Only support .jpg image sequence
        """
        _sample_metadata = (self.metadatas["RGB"])[idx]
        _ext = constant.IMGEXT
        _path = _sample_metadata.path
        _cid = _sample_metadata.cid
        _seq = video.ClippedImageSequence(
            _path, clip_len=self.clip_len, ext=_ext, **self.kwargs)
        _blob = _seq.get_varray()
        return(_blob, _cid)        


class SegmentedVideoDataset(VideoDataset):
    """

    """
    def __init__(self, root, name, seg_num,
                 split=constant.TRAINSET,
                 modalities={'RGB': constant.IMGSEQ},
                 *args, **kwargs
                ):
        """
        Only support jpg image sequence
        """   
        super(SegmentedVideoDataset, self).__init__(
            root, name, split, modalities, args, **kwargs
            )
        self.seg_num = seg_num

    def __getitem__(self, idx):
        _sample_metadata = (self.metadatas["RGB"])[idx]
        _ext = constant.IMGEXT
        _path = _sample_metadata.path
        _cid = _sample_metadata.cid
        _seq = video.SegmentedImageSequence(
            _path, seg_num=self.seg_num, ext=_ext, **self.kwargs)
        _blob = _seq.get_varray()
        return(_blob, _cid)    


def test():

    test_components = {
        'basic' : True,
        '__len__' : True,
        '__getitem__' : True
    }
    
    test_configuration = {
        'datasets'   : ["weizmann", "ucf101"]
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
            
                if (test_components['__getitem__']):
                    # print(allset.__getitem__(allset.__len__()-1))
                    for _idx in tqdm.tqdm(range(allset.__len__())):
                        allset.__getitem__(_idx)
                    for _idx in tqdm.tqdm(range(trainset.__len__())):
                        trainset.__getitem__(_idx)
                    for _idx in tqdm.tqdm(range(testset.__len__())):
                        testset.__getitem__(_idx)                                            


   



if __name__ == "__main__":

    if __test__:
        test()
