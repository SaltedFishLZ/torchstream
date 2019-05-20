# -*- coding: utf-8 -*-
## @package metadata
# Dataset Meta-data Management Module
#
#

import os
import sys
import copy
import logging
import importlib
from collections import Counter

from . import constant
from . import utilities
from .constant import __test__, __verbose__, __vverbose__, \
    __supported_dataset_styles__

__verbose__ = True
__vverbose__ = True

## Class
#  
#  ???
class Sample(object):
    """
    An video sample struct containing the meta-data of a video sample
    """

    ## Constructor
    #  
    #  @param root str: absolute root path of the dataset
    #  @param path str: absolute path of 1 video sample
    #  @param name str: file name (without any extension and path)
    #  @param ext str:  file extension (e.g., "avi", "mp4"), '.' excluded.  
    #      If it == constant.IMGSEQ, it means the sample is sliced into a 
    #      sequence of images (not limited to RGB modality) and the images 
    #      are stored in the folder which is the "path" mentioned above.
    #      NOTE: Currently, we only handle images stored in .jpg formats
    #  @param lbl str:  
    #      label of the sample, is a unique string in certain dataset
    #  @param cid str:  
    #      class id of the sample, is the numerical representation of label
    def __init__(self, root, path, name,
                 seq=True, mod="RGB", ext=constant.IMGSEQ,
                 lbl=constant.LABEL_UNKOWN, cid=constant.CID_UNKOWN):
        """
        Initailization function
        """
        self.root = root
        self.path = path
        self.name = name
        self.mod = mod
        self.ext = ext
        self.seq = seq
        self.lbl = lbl
        self.cid = cid

    ## Documentation for a method.
    #  @param self The object pointer.
    def __repr__(self, idents=0):
        string = idents * "\t" + "Sample Object: \n"
        string += idents * "\t" + str(self.name)
        if self.seq:
            string += "(frame sequence)"
        string += '\n'
        string += idents * "\t" + "[label] : {}  \t".format(self.lbl)
        string += idents * "\t" + "[cid] : {} \t".format(self.cid)
        string += idents * "\t" + "[path] : {}".format(self.path)
        return string

    ## Documentation for a method.
    #  @param self The object pointer.
    def __eq__(self, other):
        if isinstance(other, Sample):
            return(
                (self.root == other.root)
                and (self.path == other.path)
                and (self.name == other.name)
                and (self.mod == other.mod)
                and (self.ext == other.ext)
                and (self.seq == other.seq)
                and (self.lbl == other.lbl)
                and (self.cid == other.cid)
            )
        return False

    ## Documentation for a method.
    #  @param self The object pointer.
    def __ne__(self, other):
        return not self.__eq__(other)

    ## Documentation for a method.
    #  @param self The object pointer.
    def __hash__(self):
        return(hash((
            self.root,
            self.path,
            self.name,
            self.mod,
            self.ext,
            self.seq,
            self.lbl,
            self.cid
        )))

    ## Documentation for a method.
    #  @param self The object pointer.    
    def root_migrated(self, new_root):
        """
        Get a new Sample object with dataset root migrated to a new one.
        """
        rel_path = os.path.relpath(self.path, self.root)
        new_path = os.path.join(new_root, rel_path)
        return(
            Sample(
                root=new_root,
                path=new_path,
                name=self.name,
                seq=self.seq,
                mod=self.mod,
                lbl=self.lbl,
                cid=self.cid
            )
        )

    ## Doc
    #  
    def to_video(self, ext):
        self.ext = ext
        self.path = utilities.strip_extension(self.path) + "." + ext

    ## Doc
    #  @param ext: net file extension
    def to_images(self, ext):
        self.seq = True
        self.ext = ext
        self.path = utilities.strip_extension(self.path)



## SampleSet: A class containing a set of samples with some statistics
#  
#  Details :
class SampleSet(object):
    """
    A set of samples with some statistics information
    """
    ##  Constructor Function
    #  @param samples set: a set of Sample objects.
    #  @param labels set|list: a set/list of label names (str).
    #  If not specified, will count all possible labels from the samples.
    #  @param eager bool: eager execution
    def __init__(self, samples, labels=None, eager=True):
        """
        """
        self.samples = samples
        self.counts = dict()

        if labels is None:
            if eager:
                for _sample in samples:
                    _label = _sample.lbl
                    if _label is None:
                        continue
                    if _label in self.counts:
                        self.counts[_label] += 1
                    else:
                        self.counts[_label] = 1
                    self.samples.add(_sample)
        else:
            self.counts = dict.fromkeys(labels, 0)
            if eager:
                ## Get statistics immediately
                for _sample in samples:
                    _label = _sample.lbl
                    if _label is None:
                        continue
                    if _label in labels:
                        self.counts[_label] += 1
                        self.samples.add(_sample)

    ## Documentation for a method.
    #  @param self The object pointer.
    def __repr__(self, idents=0):
        string = idents * "\t" + "SampleSet Object: \n"
        for _label in self.counts:
            string += idents * "\t" + "[{}]: \t{}\n".\
                format(_label, self.counts[_label])
        return string

    ## Documentation for a method.
    #  @param self The object pointer.
    def __eq__(self, other):
        return (
            (self.samples == other.samples)
            and (self.counts == other.counts)
            )

    ## Documentation for a method.
    #  @param self The object pointer.
    def __ne__(self, other):
        return not self.__eq__(other)

    ## Documentation for a method.
    #  @param self The object pointer.
    def __hash__(self):
        return(hash((
            self.samples, 
            self.counts)))

    ## Documentation for a method.
    #  @param self The object pointer.
    def __len__(self):
        return len(self.samples)

    ## Documentation for a method.
    #  @param self The object pointer.
    def get_samples(self):
        """
        Get a list of samples from samples set
        """
        samples = list(self.samples)
        return samples

    ## Documentation for a method.
    #  @param self The object pointer.
    def get_statistics(self):
        """
        """
        return self.counts

    ## Filter samples using a filter.
    #  @param filter_ callable: a filter with input being a Sample object
    def filter_samples(self, filter_):
        """
        """
        new_samples = set()
        for _sample in self.samples:
            if not filter_(_sample):
                if __verbose__:
                    info_str = "SampleSet:[filter_samples], remove\n{}"\
                        .format(_sample)
                    logging.info(info_str)
                    if __vverbose__:
                        print(info_str)  
                if _sample.lbl != constant.LABEL_UNKOWN:
                    self.counts[_sample.lbl] -= 1
            else:
                new_samples.add(_sample)
        self.samples = new_samples

    ## Documentation for a method.
    #  @param self The object pointer.
    def update_labels(self, labels):
        """
        Update labels and corresponding counts. 
        """
        for _label in labels:
            if _label == constant.LABEL_UNKOWN:
                continue
            if _label not in self.counts:
                self.counts[_label] = 0

    ## Documentation for a method.
    #  @param self The object pointer.
    def update_samples(self, samples):
        """
        Update self.samples (add new samples) and corresponding statistics.
        NOTE: currently, we only update thoses samples with the same labels
        of existing ones.

        @param samples set: a set of Sample objects.
        """
        labels = self.counts.keys()
        for _sample in samples:
            _label = _sample.lbl
            if _label == constant.LABEL_UNKOWN:
                continue
            if _label in labels:
                self.counts[_label] += 1
                self.samples.add(_sample)  

    ## Documentation for a method.
    #  @param self The object pointer.
    def refresh_statistics(self):
        """!
        Update self.counts to make all statstistics consistent when you change
        the data without using member functions
        """
        self.counts = dict.fromkeys(self.counts.keys(), 0)
        for _sample in self.samples:
            _label = _sample.lbl
            if _label in self.counts:
                self.counts[_label] += 1
                self.samples.add(_sample)

    ## Documentation for a method.
    #  @param self The object pointer.
    def migrate_root(self, new_root):
        """
        """
        _new_samples = set()
        for _sample in self.samples:
            _new_samples.add(_sample.root_migrated(new_root))
        self.samples = _new_samples

    ## Documentation for a method.
    #  @param self The object pointer.
    def root_migrated(self, new_root):
        """
        """
        new_sample_set = copy.deepcopy(self)
        new_sample_set.migrate_root(new_root)
        return new_sample_set

## Collector
#  
#  Details
class Collector(object):
    """
    A helper functor which deals with samples' meta-data of a certain dataset.
    We only deal with meta-data in it.
    NOTE: Following the "do one thing at once" priciple, we only deal with 1 
    data type of 1 data modality in 1 collector object.
    """
    ## Documentation for a method.
    #  @param self The object pointer.
    def __init__(self, root, dset, lbls=None,
                 mod="RGB", ext=constant.IMGSEQ
                ):
        """!
        Initailization function

        @param root str: root path of the dataset
        @param dset module: meta dataset
        @param lbls set|list: a set/list of label names (str).
            If not specified, will count all possible labels from the samples.
        @param mod str: data modality
        @param ext str: file extension, "" means image(jpg) sequence
        """
        # santity check
        assert (dset.__style__ in __supported_dataset_styles__), \
            "Unsupported Dataset Struture Style"
        self.root = copy.deepcopy(root)
        self.dset = dset
        self.lbls = copy.deepcopy(lbls)
        self.mod = copy.deepcopy(mod)
        self.ext = copy.deepcopy(ext)

    ## Documentation for a method.
    #  @param self The object pointer.
    def __repr__(self, idents=0):
        string = idents * "\t" + "Meta-data Collector Object\n"
        string += idents * "\t" + "[root path] : {}\n".format(self.root)
        string += idents * "\t" + "[specified labels]: {}\n".format(self.lbls)
        string += idents * "\t" + "[modality] : {}\t".format(self.mod)
        string += idents * "\t" + "[extension] : {}\t".format(self.ext)
        return string

    ## Documentation for a method.
    #  @param self The object pointer.
    def collect_samples(self):
        """
        Collect a list of samples of given labels, given data modality and
        given file extension.

        @param return SampleSet:
            a set of Sample objects and corresponding statistics.
        """

        style = self.dset.__style__
        seq = (constant.IMGSEQ == self.ext)
        samples = set()

        ## 1. main loop
        #  get all samples' meta-data (file path, annotation, seq or not, etc)
        if "UCF101" == style:
            for _label in os.listdir(self.root):
                # bypass invalid labels
                if self.lbls is not None:
                    if _label not in self.lbls:
                        continue
                _cid = (self.dset.__labels__[_label])
                for _video in os.listdir(os.path.join(self.root, _label)):
                    # bypass invalid files
                    if self.ext not in _video:
                        continue
                    _path = os.path.join(self.root, _label, _video)
                    if not seq:
                        _name = utilities.strip_extension(_video)
                    else:
                        _name = _video
                    _sample = Sample(root=self.root, path=_path, name=_name,
                                     seq=seq, mod=self.mod, ext=self.ext,
                                     lbl=_label, cid=_cid)
                    samples.add(_sample)
        ## 
        #  
        #  
        elif "20BN" == style:
            for _video in os.listdir(self.root):
                # bypass invalid files
                if self.ext not in _video:
                    continue
                if not seq:
                    _name = utilities.strip_extension(_video)
                else:
                    _name = _video

                _label = self.dset.__targets__[_video]
                if self.lbls is not None:
                    if _label not in self.lbls:
                        continue
                if _label == constant.LABEL_UNKOWN:
                    _cid = constant.CID_UNKOWN
                else:
                    _cid = self.dset.__labels__[_label]

                _path = os.path.join(self.root, _video)
                _sample = Sample(root=self.root, path=_path, name=_name,
                                 seq=seq, mod=self.mod, ext=self.ext,
                                 lbl=_label, cid=_cid)
                samples.add(_sample)
        ## 
        #  
        else:
            raise Exception("Unsupported Dataset Style: {}".format(style))

        # output status
        if __verbose__:
            info_str = "Collector: [collect_samples] get {} samples."\
                .format(len(samples))
            if __vverbose__:
                print(info_str)

        ## 2. get statistics
        # count corresponding sample number for each label 
        ret = SampleSet(samples, self.lbls)

        return ret

    ## Documentation for a method.
    #  @param self The object pointer.
    def __call__(self):
        return self.collect_samples()

    ## Documentation for a method.
    #  @param self The object pointer.
    def check_integrity(self, lbls=None, sample_set=None):
        """
        Check meta-data integrity
        """
        if sample_set is None:
            sample_set = self.collect_samples()

        passed = True
        warn_str = "Integrity check failed.\n"

        # check labels
        if lbls is None:
            labels_got = set(sample_set.counts.keys())
            labels_expected = set(self.dset.__labels__.keys())
            if (labels_got != labels_expected):
                warn_str += "label mismatch"
                passed = False

        # check sample number for each class
        for _label in sample_set.counts:
            _sample_count = sample_set.counts[_label]
            ref_sample_count = self.dset.__sample_num_per_class__[_label]

            # reference sample number is an interval
            if isinstance(ref_sample_count, list):
                if not ((_sample_count >= ref_sample_count[0])
                        and (_sample_count <= ref_sample_count[1])):
                    fmt_str = "[{}]:\t sample number mismatch, a number in "
                    fmt_str += "{} expected, [{}] got.\n"
                    warn_str += fmt_str.\
                        format(_label, ref_sample_count, _sample_count)
                    passed = False
            # reference sample number is an exact number
            elif isinstance(ref_sample_count, int):
                if ref_sample_count != _sample_count:
                    fmt_str = "[{}]:\t sample number mismatch, [{}] expected, "
                    fmt_str += "[{}] got.\n"
                    warn_str += fmt_str.\
                        format(_label, ref_sample_count, _sample_count)
                    passed = False
            else:
                raise Exception("Incorrect reference sample number type")

        if not passed:
            logging.warn(warn_str)
            return False

        return True


if __name__ == "__main__":

    DATASET = "sth_sth_v1"
    dataset_mod = importlib.import_module("vdataset.{}".format(DATASET))

    lbls=dataset_mod.__labels__

    collector = Collector(
        dataset_mod.raw_data_path,
        dataset_mod,
        # lbls={'Sliding Two Fingers Up',},
        ext=constant.IMGSEQ
        )
    
    sample_set = collector()
    # for _sample in sample_set.samples:
    #     if ".avi" not in _sample.path:
    #         print(_sample)
    print(len(sample_set.samples))
    print(sample_set.counts)
    # print(collector.check_integrity())
    print(sample_set.get_samples()[1])

