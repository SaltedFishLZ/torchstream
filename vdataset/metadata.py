# -*- coding: utf-8 -*-
# Dataset Meta Data Management Module
#
#
__test__    =   True
__strict__  =   True
__verbose__ =   True
__vverbose__=   False

import os
import sys
import copy
import logging
import importlib

import video
import UCF101, HMDB51, Weizmann

from __init__ import __supported_dataset_styles__, __supported_datasets__


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
    def __init__(self, root, style, label_map, 
                mod = "RGB", ext= "avi", 
                seek_file=True, part=None):
        '''
        part = None: all data are collected
        '''
        # santity check
        assert (style in __supported_dataset_styles__), \
            "Unsupported Dataset Struture Style"
        self.root = copy.deepcopy(root)
        self.style = copy.deepcopy(style)
        self.label_map = copy.deepcopy(label_map)
        self.mod = copy.deepcopy(mod)
        self.ext = copy.deepcopy(ext)

        self.labels = []
        self.samples = []
        if (True == seek_file):
            labels, samples = self.collect_samples(self.root, self.style, 
                self.label_map, self.mod, self.ext)
            # NOTE: here we use list.extend !!!
            self.labels.extend(labels)
            self.samples.extend(samples)

    @staticmethod
    def collect_samples(root, style, label_map, mod, ext):
        '''
        collect a list of samples = list of samples, while each sample
        = (video, relative path, class id, label)
        here, video may refers to a collection of images
        '''
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
                    _path = os.path.join(root, _label, _video)
                    _sample = Sample(_path, _video, ext, 
                                  _label, label_map[_label])
                    samples.append(copy.deepcopy(_sample))
            # return results
            return(labels, samples)               
        else:
            assert True, "Unsupported Dataset Struture Style"        

    def __filter_samples__(self, sample_filter):
        filtered_samples = []
        for _sample in self.samples:
            if (sample_filter(_sample)):
                filtered_samples.append(_sample)
            else:
                # DEBUG
                print(_sample)
                pass
        self.samples = filtered_samples

    def __get_samples__(self):
        return(self.samples)

    def __check_integrity__(self, dataset_mod):
        # assert (dataset in __supported_datasets__), "Unsupported Dataset"
        # dataset_mod = importlib.import_module(".{}".format(DATASET))
        
        # check class number
        if (sorted(self.labels) != sorted(dataset_mod.__classes__)):
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
            ref_sample_count = dataset_mod.__samples__[_label]
            if (type(ref_sample_count) == list):
                if (not (_sample_count >= ref_sample_count[0])
                    and (_sample_count <= ref_sample_count[1]) ):
                    passed = False
            elif (type(ref_sample_count) == int):
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

    DATASET = "HMDB51"
    dataset_mod = importlib.import_module("{}".format(DATASET))

    collector = VideoCollector(
        dataset_mod.raw_data_path,
        __supported_datasets__[DATASET],
        dataset_mod.label_map
        )
    
    for _sample in collector.__get_samples__():
        print(_sample) 

    check = collector.__check_integrity__(dataset_mod)  
    if (check):
        print("passed")
