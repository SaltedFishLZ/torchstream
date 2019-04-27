# -*- coding: utf-8 -*-
import os
import sys
import copy
import logging
import importlib

import torch
import torch.utils.data as torchdata

from .__init__ import __test__, __strict__, __verbose__, __vverbose__, \
    __supported_datasets__, __supported_styles__, __supported_modalities__, \
    __supported_modality_files__
    
from . import video, metadata



class VideoDataset(torchdata.Dataset):
    '''
    NOTE: TODO: Currently, This shall be an abstract base class.
    It should never be used in deployment !!!
    '''
    def __init__(self, root, dataset, part = None, split="1",
        modalities = {'RGB': ['jpg']}):
        # TODO: support for raw videos!
        # currently we only support read processed images
        # TODO: support multiple input data modalities

        # santity check
        assert os.path.exists(root), "Dataset path not exists"
        assert dataset in __supported_datasets__, "Unsupported Dataset"
        assert (1==len(modalities)), "Only support 1 data modality now"
        for _mod in modalities:
            assert _mod in __supported_modalities__, 'Unsupported Modality'
            print(__supported_modality_files__[_mod])
            for _ext in modalities[_mod]:
                assert _ext in __supported_modality_files__[_mod],\
                    ("Unspported input file type: {} for modality: {}"\
                    .format(_ext, _mod))
       
        self.root = copy.deepcopy(root)
        self.dataset = copy.deepcopy(dataset)
        self.part = copy.deepcopy(part)
        self.split = split
        self.modalities = copy.deepcopy(modalities)

        # collect metadata
        self.style = __supported_datasets__[self.dataset]
        self.dset = importlib.import_module("vdataset.{}".format(dataset))
        self.labels = self.dset.__labels__
        self.mcollect = metadata.VideoCollector(
            root = self.root, dset = self.dset,
            # currently only support RGB modality and sliced pictures
            mod = "RGB", ext = self.modalities["RGB"][0], part=self.part
        )

        # filter samples
        if (self.part != None):
            if (self.part == "train"):
                sample_filter = self.dset.TrainsetFilter(split=self.split)
            elif (self.part == "val"):
                sample_filter = self.dset.ValsetFilter(split=self.split)
            elif (self.part == "test"):
                sample_filter = self.dset.TestsetFilter(split=self.split)
            else:
                assert True, "?"
            # filtering data
            self.mcollect._filter_samples_(sample_filter)

    def __len__(self):
        return(len(self.mcollect.samples))

    def __getitem__(self, idx):
        # if (__strict__):
        #     assert True, \
        #         "VideoDataset is abstract, __getitem__ must be overrided"
        # NOTE
        # currently, we intended to return a Numpy ndarray while it 
        # may consume too much memory.

        # get sample metadata
        _sample_metadata = self.mcollect._get_samples_()[idx]
        # load data according to metadata
        # NOTE: TODO: Currently, we only support sliced image sequence as 
        # input.You cannot load a video file directly, sorry for that.
        _ext = _sample_metadata.ext
        assert (_ext == "jpg"), "Currently, only support RGB data in .jpg"
        _path = _sample_metadata.path
        _cid = _sample_metadata.cid
        _seq = video.ImageSequence(_path, ext=_ext)
        # get all frames as a vaaray
        _blob = _seq._get_varray_(list())
        # return (a [T][H][W][C] ndarray, class id)
        # ndarray may need to be converted to [T][C][H][W] format in PyTorch
        return(_blob, _cid)



class ClippedVideoDataset(VideoDataset):
    '''
    '''
    def __init__(self, root, dataset, clip_len, part = None, split="1",
        modalities = {'RGB': ['jpg']}):
        super(ClippedVideoDataset, self).__init__(\
            root, dataset, part, split, modalities)
        self.clip_len = copy.deepcopy(clip_len)

    def __getitem__(self, idx):
        _sample_metadata = self.mcollect._get_samples_()[idx]
        _ext = _sample_metadata.ext
        assert (_ext == "jpg"), "Currently, only support RGB data in .jpg"
        _path = _sample_metadata.path
        _cid = _sample_metadata.cid
        _seq = video.ClippedImageSequence(_path, clip_len=self.clip_len, ext=_ext)
        _blob = _seq._get_varray_()
        return(_blob, _cid)        

class SegmentedVideoDataset(VideoDataset):
    def __init__(self, root, dataset, seg_num, part = None, split="1",
        modalities = {'RGB': ['jpg']}):
        super(SegmentedVideoDataset, self).__init__(\
            root, dataset, part, split, modalities)
        self.seg_num = copy.deepcopy(seg_num)

    def __getitem__(self, idx):
        _sample_metadata = self.mcollect._get_samples_()[idx]
        _ext = _sample_metadata.ext
        assert (_ext == "jpg"), "Currently, only support RGB data in .jpg"
        _path = _sample_metadata.path
        _cid = _sample_metadata.cid
        _seq = video.SegmentedImageSequence(_path, seg_num=self.seg_num, ext=_ext)
        _blob = _seq._get_varray_()
        return(_blob, _cid)    


if __name__ == "__main__":

    if (__test__):

        test_components = {
            'basic':True,
            '__len__':True,
            '__getitem__':True
        }
        
        test_configuration = {
            'datasets'   : ["Weizmann",]
        }

        for DATASET in (test_configuration['datasets']):
            if (test_components['basic']):

                dset = importlib.import_module(
                    "vdataset.{}".format(DATASET))
                allset = ClippedVideoDataset(
                    dset.prc_data_path,DATASET,clip_len=4,split="1")
                trainset = ClippedVideoDataset(
                    dset.prc_data_path,DATASET,clip_len=4,part="train",split="1")
                testset = ClippedVideoDataset(
                    dset.prc_data_path,DATASET,clip_len=4,part="test",split="1")
            
                if (test_components['__len__']):
                    print("All samples number:")
                    print(allset.__len__())
                    print("Training Set samples number:")
                    print(trainset.__len__())
                    print("Testing Set samples number:")
                    print(testset.__len__())
                
                    if (test_components['__getitem__']):
                        for _idx in range(allset.__len__()):
                            allset.__getitem__(_idx)
                        for _idx in range(trainset.__len__()):
                            trainset.__getitem__(_idx)
                        for _idx in range(testset.__len__()):
                            testset.__getitem__(_idx)                                            


   
