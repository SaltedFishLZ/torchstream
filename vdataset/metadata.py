# -*- coding: utf-8 -*-
# Dataset Meta Data Management Module
#
#

import os
import sys
import copy
import logging
import importlib

from . import video
from .__init__ import __supported_dataset_styles__, __supported_datasets__, \
    __test__, __strict__, __verbose__, __vverbose__


class Sample(object):
    '''
    An video sample struct containing the meta-data of a video sample
    '''
    def __init__(self, path, name, ext, lbl="default", cid=-1):
        '''
        path: compelete sample path
        name: file name (without any extension and path)
        ext:  file extension (e.g., avi, mp4), '.' excluded, if it == None
              or "", it means the sample is sliced into several images (not
              limited to RGB modality) and the images are stored in the folder
              which is the "path" mentioned above.
        lbl:  label of the sample, is a unique string in certain dataset
        cid:  class id of the sample, is the numerical representation of label
        '''
        self.path   =   copy.deepcopy(path)
        self.name   =   copy.deepcopy(name)
        self.ext    =   copy.deepcopy(ext)
        self.lbl    =   copy.deepcopy(lbl)
        self.cid    =   copy.deepcopy(cid)

    def __repr__(self):
        string = str(self.name) + '\n'
        string += "[label] : {}  \t".format(self.lbl)
        string += "[cid] : {} \t".format(self.cid)
        string += "[path] : {}".format(self.path)
        return(string)

class VideoCollector(object):
    '''
    A helper class which maintains all sample's meta-data of a certain dataset
    We only deal with Meta-data in it
    TODO: support multiple data format and multiple input modality
    '''
    def __init__(self, root, dset, 
                mod = "RGB", ext= "avi", 
                seek_file=True, part=None):
        '''
        part = None: all data are collected
        '''
        # santity check
        assert (dset.__style__ in __supported_dataset_styles__), \
            "Unsupported Dataset Struture Style"
        self.root = copy.deepcopy(root)
        self.dset = dset
        self.mod = copy.deepcopy(mod)
        self.ext = copy.deepcopy(ext)

        self.labels = []
        self.samples = []
        if (True == seek_file):
            samples, labels = self.collect_samples(self.root, self.dset
                , self.mod, self.ext)
            # NOTE: here we use list.extend !!!
            self.labels.extend(labels)
            self.samples.extend(samples)
        
        if (__verbose__):
            info_str = "VideoCollector initialized: "
            info_str += "paths: {}; style: {}; modality {}; file \
                extension {};".format(self.root, self.dset.__style__, 
                self.mod, self.ext)
            logging.info(info_str)
            if (__vverbose__):
                print(info_str)

    @staticmethod
    def collect_samples(root, dset, mod, ext):
        '''
        Collect a list of samples.
        - root : root path of the dataset
        - dset : dataset as a Python module
        - mod : data modalities
        - ext : file extension
        '''
        style = dset.__style__

        if ("UCF101" == style):
            # get all labels
            labels = []
            for _label in os.listdir(root):
                if (os.path.isdir(os.path.join(root, _label))):
                    labels.append(_label)

            # TODO: check whether we need to sort it
            labels = sorted(labels)
             
            # get all videos' file paths and annotations
            # NOTE: each sample is a tuple = 
            # (vid_path, relative_addr, class_id, label)
            samples = []
            for _label in labels:
                for _video in os.listdir(os.path.join(root, _label)):
                    if (ext not in [None, ""]):
                        # split file extension
                        _name = _video[:-(len(ext)+1)]
                    else:
                        _name = copy.deepcopy(_video)
                    _path = os.path.join(root, _label, _video)
                    _sample = Sample(_path, _name, ext, 
                                  _label, dset.__labels__[_label])
                    samples.append(copy.deepcopy(_sample))

            # output status
            if (__verbose__):
                info_str = "[collect_samples] get {} samples"\
                    .format(samples)
                if (__vverbose__):
                    print(info_str)
            return(samples, labels)
 
        else:
            assert True, "Unsupported Dataset Struture Style"        

    def _filter_samples_(self, sample_filter):
        filtered_samples = []
        for _sample in self.samples:
            if (sample_filter(_sample)):
                filtered_samples.append(_sample)
            else:
                pass
        self.samples = filtered_samples

    def _get_samples_(self):
        return(self.samples)

    def _check_integrity_(self, dset):
        
        # check class number
        if (sorted(self.labels) != sorted(dset.__labels__.keys())):
            warn_str = "Integrity check failed, "
            warn_str += "class numbver mismatch"
            logging.warn(warn_str)
            return False
        
        # check sample number
        # count samples for each class
        _sample_count_dict = dict(zip(self.labels, len(self.labels)*[0]))
        for _sample in self.samples:
            _sample_count_dict[_sample.lbl] += 1
        passed = True
        # check sample number for each class
        for _label in self.labels:
            _sample_count = _sample_count_dict[_label]
            ref_sample_count = dset.__sample_num_per_class__[_label]

            if (type(ref_sample_count) == list):
                # reference sample number is an interval
                if (not (_sample_count >= ref_sample_count[0])
                    and (_sample_count <= ref_sample_count[1]) ):
                    passed = False
            elif (type(ref_sample_count) == int):
                # reference sample number is exact
                if (ref_sample_count != _sample_count):
                    passed = False
            else:
                assert True, "Incorrect reference sample number"
            if (not passed):
                warn_str = "Integrity check failed, "
                warn_str += "sample numbver mismatch for class [{}]".\
                    format(_label)
                logging.warn(warn_str)
                break

        return True



if __name__ == "__main__":

    DATASET = "UCF101"
    dataset_mod = importlib.import_module("vdataset.{}".format(DATASET))

    collector = VideoCollector(
        dataset_mod.prc_data_path,
        dataset_mod,
        ext=""
        )
    
    for _sample in collector._get_samples_():
        print(_sample) 

    check = collector._check_integrity_(dataset_mod)  
    if (check):
        print("passed")
