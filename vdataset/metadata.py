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
from .__init__ import *

__verbose__ = True
__vverbose__ = True

class Sample(object):
    '''
    An video sample struct containing the meta-data of a video sample
    '''
    def __init__(self, path, name, seq=True, ext="jpg", lbl=None, cid=-1):
        '''
        path: absolute path of the video sample
        name: file name (without any extension and path)
        ext:  file extension (e.g., avi, mp4), '.' excluded, if it == None
              or "", it means the sample is sliced into several images (not
              limited to RGB modality) and the images are stored in the folder
              which is the "path" mentioned above.
              NOTE: Currently, we only handle images stored in .jpg formats
        lbl:  label of the sample, is a unique string in certain dataset
        cid:  class id of the sample, is the numerical representation of label
        '''
        self.path   =   copy.deepcopy(path)
        self.name   =   copy.deepcopy(name)
        self.ext    =   copy.deepcopy(ext)
        self.seq    =   copy.deepcopy(seq)
        self.lbl    =   copy.deepcopy(lbl)
        self.cid    =   copy.deepcopy(cid)

    def __repr__(self):
        string = str(self.name)
        if (self.seq):
            string += "(frame sequence)"
        string += '\n'
        string += "[label] : {}  \t".format(self.lbl)
        string += "[cid] : {} \t".format(self.cid)
        string += "[path] : {}".format(self.path)
        return(string)


class Collector(object):
    '''
    A helper class which maintains samples' meta-data of a certain dataset.
    We only deal with Meta-data in it.
    NOTE: Following the "do one thing at once" priciple, we only deal with 1 
    data type of 1 data modality in 1 collector object.
    TODO: enable file name template
    '''
    def __init__(self, root, dset, 
                mod="RGB", ext="", 
                eager=True):
        """!

        @param root str: root path of the dataset        
        @param dset Python module: meta dataset
        @param mod str: data modality
        @param ext str: file extension, "" means image(jpg) sequence
        @param eager bool: collecting data eagerly in initialization
        """
        # santity check
        assert (dset.__style__ in __supported_dataset_styles__), \
            "Unsupported Dataset Struture Style"
        self.root = copy.deepcopy(root)
        self.dset = dset
        self.mod = copy.deepcopy(mod)
        self.ext = copy.deepcopy(ext)

        self.labels = set()
        self.samples = set()
        if (True == eager):
            samples, labels = self.collect_samples(root=self.root,
                dset=self.dset, mod=self.mod, ext=self.ext)
            # use list.extend to update labels and samples
            self.labels.update(labels)
            self.samples.update(samples)
        
        if (__verbose__):
            info_str = "VideoCollector initialized: "
            info_str += "paths: {}; style: {}; modality {}; file \
                extension {};".format(self.root, self.dset.__style__, 
                self.mod, self.ext)
            logging.info(info_str)
            if (__vverbose__):
                print(info_str)

    @staticmethod
    def collect_labels(root, dset, mod, ext):
        """!
        Collect all possible labels from a certain part (modality & extension)
        of a dataset.   

        @param root str: root path of the dataset        
        @param dset Python module: meta dataset
        @param mod str: data modality
        @param ext str: file extension
        @param return set: a set of all labels appeared in certain part of the
        dataset.
        """
        style = dset.__style__
        seq = ("" == ext)

        if ("UCF101" == style):

            labels = set()
            for _label in os.listdir(root):
                if (os.path.isdir(os.path.join(root, _label))):
                    labels.add(_label)

            # TODO: check whether we need to sort it
            labels = sorted(labels)
            
            # output status
            if (__verbose__):
                info_str = "VideoCollector: [collect_samples] get {} labels"\
                    .format(len(labels))
                if (__vverbose__):
                    print(info_str)
            
            return(labels)
        
        else:
            assert True, NotImplementedError
            exit(1)


    @staticmethod
    def collect_samples(root, dset, mod, ext):
        """!
        Collect a list of samples.    

        @param root str: root path of the dataset        
        @param dset Python module: meta dataset
        @param mod str: data modality
        @param ext str: file extension
        @param return tuple: (samples, labels); samples is a list of Sample
        objects, labels is a list of strs;
        """

        style = dset.__style__
        seq = ("" == ext)

        # get all samples' meta-data (file path, annotation, seq or not, etc)
        if ("UCF101" == style):
            
            samples = set()
            for _label in os.listdir(root):
                for _video in os.listdir(os.path.join(root, _label)):
                    if (False == seq):
                        # split file extension
                        _name = _video[:-(len(ext)+1)]
                    else:
                        _name = copy.deepcopy(_video)
                    _path = os.path.join(root, _label, _video)
                    _sample = Sample(path=_path, name=_name, seq=seq, ext=ext, 
                            lbl=_label, cid=(dset.__labels__[_label]))
                    samples.add(_sample)

            # output status
            if (__verbose__):
                info_str = "VideoCollector: [collect_samples] get {} samples"\
                    .format(len(samples))
                if (__vverbose__):
                    print(info_str)
            return(samples)
 
        else:
            print(style)
            assert True, "Unsupported Dataset Struture Style"        

    def _filter_samples_(self, sample_filter):
        filtered_samples = []
        for _sample in self.samples:
            if (sample_filter(_sample)):
                filtered_samples.append(_sample)
            else:
                if (__verbose__):
                    info_str = "VideoCollector:[_filter_samples_], remove\n{}"\
                        .format(_sample)
                    logging.info(info_str)
                    if (__vverbose__):
                        print(info_str)  
                
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

    DATASET = "Jester"
    dataset_mod = importlib.import_module("vdataset.{}".format(DATASET))

    collector = Collector(
        dataset_mod.raw_data_path,
        dataset_mod,
        ext=""
        )
    
    for _sample in collector._get_samples_():
        print(_sample)

    check = collector._check_integrity_(dataset_mod)  
    if (check):
        print("passed")
