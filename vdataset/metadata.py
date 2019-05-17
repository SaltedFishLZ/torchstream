# -*- coding: utf-8 -*-
## @module metadata
# Dataset Meta-data Management Module
#
#

import os
import sys
import copy
import logging
import importlib

from . import video
from . import constant
from . import utilities
from .__init__ import __supported_dataset_styles__
from .__init__ import *

__verbose__ = True
__vverbose__ = True

class Sample(object):
    """!
    An video sample struct containing the meta-data of a video sample
    """
    def __init__(self, path, name, seq=True, mod="RGB", ext=constant.IMGSEQ, 
        lbl=None, cid=-1):
        """!
        Initailization function

        @param path str: absolute path of 1 video sample
        @param name str: file name (without any extension and path)
        @param ext str:  file extension (e.g., "avi", "mp4"), '.' excluded.  
            If it == constant.IMGSEQ, it means the sample is sliced into a 
            sequence of images (not limited to RGB modality) and the images 
            are stored in the folder which is the "path" mentioned above.
            NOTE: Currently, we only handle images stored in .jpg formats
        @param lbl str:  
            label of the sample, is a unique string in certain dataset
        @param cid str:  
            class id of the sample, is the numerical representation of label
        """
        self.path   =   copy.deepcopy(path)
        self.name   =   copy.deepcopy(name)
        self.mod    =   copy.deepcopy(mod)
        self.ext    =   copy.deepcopy(ext)
        self.seq    =   copy.deepcopy(seq)
        self.lbl    =   copy.deepcopy(lbl)
        self.cid    =   copy.deepcopy(cid)

    def __eq__(self, other):
        if (isinstance(other, Sample)):
            return((self.path == other.path)
                and (self.name == other.name)
                and (self.mod == other.mod)
                and (self.ext == other.ext)
                and (self.seq == other.seq)
                and (self.lbl == other.lbl)
                and (self.cid == other.cid)
            )
        else:
            return False
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return(hash(
            self.path,
            self.name,
            self.mod,
            self.ext,
            self.seq,
            self.lbl,
            self.cid
        ))

    def __repr__(self):
        string = "Sample Object: " + str(self.name)
        if (self.seq):
            string += "(frame sequence)"
        string += '\n'
        string += "[label] : {}  \t".format(self.lbl)
        string += "[cid] : {} \t".format(self.cid)
        string += "[path] : {}".format(self.path)
        return(string)


class SampleSet(object):
    """!
    A set of samples with some statistics information
    """
    def __init__(self, samples=set(), labels=None, eager=True):
        """!
        Pick samples of given labels and get corresponding statistics.
        
        @param samples set: a set of Sample objects.
        @param labels set|list: a set/list of label names (str).
            If not specified, will count all possible labels from the samples.
        @param eager bool: eager execution
        """
        self.samples = samples
        self.counts = dict()

        if (labels is None):
            if (eager):
                for _sample in samples:
                    _label = _sample.lbl
                    if (_label in labels):
                        self.counts[_label] += 1
                    else:
                        self.counts[_label] = 1
                    self.samples.add(_sample)
        else:
            self.counts = dict.fromkeys(self.counts.keys(), 0)
            if (eager):
                ## Get statistics immediately
                for _sample in samples:
                    _label = _sample.lbl
                    if (_label in labels):
                        self.counts[_label] += 1
                        self.samples.add(_sample)

    def get_samples(self):
        return(self.samples)
    
    def get_statistics(self):
        return(self.counts)

    def filter_samples(self, _f):
        """!
        Filter self.samples using a filter _f.

        @param _f callable: a filter to filter samples
        """
        for _sample in self.samples:
            if (not _f(_sample)):
                if (__verbose__):
                    info_str = "SampleSet:[filter_samples], remove\n{}"\
                        .format(_sample)
                    logging.info(info_str)
                    if (__vverbose__):
                        print(info_str)  
                self.samples.remove(_sample)
                self.counts[_sample.lbl] -= 1

    def update_labels(self, labels):
        """!
        Update labels and corresponding counts. 

        @param labels set|list: a set/list of label names (str)
        """
        for _label in labels:
            if (_label not in self.counts):
                self.counts[_label] = 0

    def update_samples(self, samples):
        """!
        Update self.samples (add new samples) and corresponding statistics.
        NOTE: currently, we only update thoses samples with the same labels
        of existing ones.

        @param samples set: a set of Sample objects.
        """
        labels = self.counts.keys()
        for _sample in samples:
            _label = _sample.lbl
            if (_label in labels):
                self.counts[_label] += 1
                self.samples.add(_sample)  

    def refresh_counts(self):
        """!
        Update self.counts to make all statstistics consistent when you change
        the data without using member functions
        """
        self.counts = dict.fromkeys(self.counts.keys(), 0)
        for _sample in self.samples:
            _label = _sample.lbl
            if (_label in self.counts):
                self.counts[_label] += 1
                self.samples.add(_sample)

    def __eq__(self, other):
        return ((self.samples == other.samples)
            and (self.counts == other.counts))

    def __ne__(self, other):
        return(not self.__eq__(other))

    def __hash__(self):
        return(hash(
            self.samples, 
            self.counts))

    def __repr__(self):
        string = "SampleSet Object: \n"
        for _label in self.counts.keys():
            string += "[{}]: \t{}\n".format(_label, self.counts[_label])
        return(string)


class Collector(object):
    '''
    A helper functor which deals with samples' meta-data of a certain dataset.
    We only deal with Meta-data in it.
    NOTE: Following the "do one thing at once" priciple, we only deal with 1 
    data type of 1 data modality in 1 collector object.
    TODO: enable file name template
    '''
    def __init__(self, root, dset, lbls=None,
                mod="RGB", ext=constant.IMGSEQ, 
                eager=True):
        """!
        Initailization function

        @param root str: root path of the dataset        
        @param dset module: meta dataset
        @param lbls set|list: a set/list of label names (str).
            If not specified, will count all possible labels from the samples.
        @param mod str: data modality
        @param ext str: file extension, "" means image(jpg) sequence
        @param eager bool: collecting data eagerly in initialization
        """
        # santity check
        assert (dset.__style__ in __supported_dataset_styles__), \
            "Unsupported Dataset Struture Style"
        self.root = copy.deepcopy(root)
        self.dset = dset
        self.lbls = copy.deepcopy(lbls)
        self.mod = copy.deepcopy(mod)
        self.ext = copy.deepcopy(ext)

    def __repr__(self):
        string = "Meta-data Collector"
        string += "\n"
        string += "[root path] : {}\n".format(self.root)
        string += "[specified labels]: {}\n".format(self.lbls)
        string += "[modality] : {}\t".format(self.mod)
        string += "[extension] : {}\t".format(self.ext)
        return(string)

    # def collect_labels(self):
    #     """!
    #     Collect all possible labels from a certain part (modality & extension)
    #     of a dataset.   

    #     @param return set: a set of all labels appeared in certain part of the
    #     dataset.
    #     """
    #     style = self.dset.__style__
    #     seq = (constant.IMGSEQ == self.ext)

    #     # @var labels
    #     labels = set()
    #     if ("UCF101" == style):
    #         for _label in os.listdir(self.root):
    #             if (os.path.isdir(os.path.join(self.root, _label))):
    #                 labels.add(_label)
    #     else:
    #         assert True, NotImplementedError
    #         exit(1)

    #     # output status
    #     if (__verbose__):
    #         info_str = "Collector: [collect_samples] get {} labels"\
    #             .format(len(labels))
    #         if (__vverbose__):
    #             print(info_str)
        
    #     return(labels)
      
    def collect_samples(self):
        """!
        Collect a list of samples of given labels, given data modality and
        given file extension.    

        @param return SampleSet:  
            a set of Sample objects and corresponding statistics.
        """

        style = self.dset.__style__
        seq = (constant.IMGSEQ == self.ext)
        samples = set()
        
        ## 1. main loop
        # get all samples' meta-data (file path, annotation, seq or not, etc)
        if ("UCF101" == style):
            for _label in os.listdir(self.root):
                _cid = (self.dset.__labels__[_label])
                for _video in os.listdir(os.path.join(self.root, _label)):
                    if (False == seq):
                        _name = utilities.strip_extension(_video)
                    else:
                        _name = copy.deepcopy(_video)
                    _path = os.path.join(self.root, _label, _video)
                    _sample = Sample(path=_path, name=_name, seq=seq,
                            mod=self.mod, ext=self.ext, 
                            lbl=_label, cid=_cid)
                    samples.add(_sample)
        else:
            print(style)
            assert True, "Unsupported Dataset Struture Style"        

        # output status
        if (__verbose__):
            info_str = "Collector: [collect_samples] get {} samples."\
                .format(len(samples))
            if (__vverbose__):
                print(info_str)

        ## 2. get statistics
        # count corresponding sample number for each label 
        ret = SampleSet(samples, self.lbls)
        
        return(ret)

    def __call__(self):
        return(self.collect_samples())

    # def _check_integrity_(self, dset):
        
    #     # check class number
    #     if (sorted(self.labels) != sorted(dset.__labels__.keys())):
    #         warn_str = "Integrity check failed, "
    #         warn_str += "class numbver mismatch"
    #         logging.warn(warn_str)
    #         return False
        
    #     # check sample number
    #     # count samples for each class
    #     _sample_count_dict = dict(zip(self.labels, len(self.labels)*[0]))
    #     for _sample in self.samples:
    #         _sample_count_dict[_sample.lbl] += 1
    #     passed = True
    #     # check sample number for each class
    #     for _label in self.labels:
    #         _sample_count = _sample_count_dict[_label]
    #         ref_sample_count = dset.__sample_num_per_class__[_label]

    #         if (type(ref_sample_count) == list):
    #             # reference sample number is an interval
    #             if (not (_sample_count >= ref_sample_count[0])
    #                 and (_sample_count <= ref_sample_count[1]) ):
    #                 passed = False
    #         elif (type(ref_sample_count) == int):
    #             # reference sample number is exact
    #             if (ref_sample_count != _sample_count):
    #                 passed = False
    #         else:
    #             assert True, "Incorrect reference sample number"
    #         if (not passed):
    #             warn_str = "Integrity check failed, "
    #             warn_str += "sample numbver mismatch for class [{}]".\
    #                 format(_label)
    #             logging.warn(warn_str)
    #             break

    #     return True



if __name__ == "__main__":

    DATASET = "Jester"
    dataset_mod = importlib.import_module("vdataset.{}".format(DATASET))

    collector = Collector(
        dataset_mod.raw_data_path,
        dataset_mod,
        ext=constant.IMGSEQ
        )
    
    sample_set = collector()
    for _sample in sample_set.samples:
        print(_sample)


