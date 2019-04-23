# -*- coding: utf-8 -*-
__test__    =   True
__strict__  =   True
__verbose__ =   True
__vverbose__=   True

import os
import sys
import copy
import logging
import importlib

import torch
import torch.utils.data as torchdata

from .__init__ import __supported_datasets__, __supported_dataset_styles__, \
    __supported_modalities__, __supported_modality_files__
import video, metadata



class VideoDataset(torchdata.Dataset):
    '''
    This shall be an abstract base class. It should never be used in deployment.
    '''
    def __init__(self, root, dataset, part = None,
        modalities = {'RGB': ['jpg']}
        ):
        # TODO: support for raw videos!
        # currently we only support read processed images

        # santity check
        assert os.path.exists(root), "Dataset path not exists"
        assert dataset in __supported_datasets__, "Unsupported Dataset"
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
        self.modalities = copy.deepcopy(modalities)

        # collect metadata
        self.dataset_style = __supported_datasets__[self.dataset]
        self.dataset_mod = importlib.import_module("{}".format(dataset))
        self.label_map = self.dataset_mod.label_map
        self.metadata_collector = metadata.VideoCollector(
            root = self.root, style = self.dataset_style,
            label_map = self.label_map,
            # currently only support 1 modality and sliced pictures
            mod = "RGB", ext = "", part=self.part
        )

        # filter samples
        if (self.part != None):
            if (self.part == "train"):
                sample_filter = self.dataset_mod.for_train()
            elif (self.part == "val"):
                sample_filter = self.dataset_mod.for_val()
            elif (self.part == "test"):
                sample_filter = self.dataset_mod.for_test()
            else:
                assert True, "?"
            # filtering data
            self.metadata_collector.__filter_samples__(sample_filter)

    def __len__(self):
        return(len(self.metadata_collector.samples))

    def __getitem__(self, idx):
        assert True, "VideoDataset is abstract, __getitem__ must be overrided"



if __name__ == "__main__":

    DATASET = "UCF101"
    dataset_mod = importlib.import_module("{}".format(DATASET))

    allset = VideoDataset(
        dataset_mod.prc_data_path, DATASET)
    print(allset.__len__())

    trainset = VideoDataset(
        dataset_mod.prc_data_path, DATASET, part="train")
    print(trainset.__len__())

