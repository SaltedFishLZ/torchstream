"""Helper functions for collecting metadata from a dataset directory
"""
__all__ = [
    "collect_ucf_samples"
]

import os
import copy
import logging

from . import __config__
from .sample import Sample
from ..utils.filesys import strip_extension
from .__support__ import __SUPPORTED_MODALITIES__, __SUPPORTED_IMAGES__
from .__support__ import __SUPPORTED_LAYOUTS__

# ---------------------------------------------------------------- #
#                  Configuring Python Logger                       #
# ---------------------------------------------------------------- #

if __config__.__VERY_VERBOSE__:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s"
    )
elif __config__.__VERY_VERBOSE__:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s - %(levelname)s - %(message)s"
    )
elif __config__.__VERBOSE__:
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
else:
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- #
#            Collect Samples and Get A Set of Samples              #
# ---------------------------------------------------------------- #

def collect_samples_ucf(root, lbls, mod, ext,
                        **kwargs):
    """ Collect samples from a dataset with a 20BN style layout
    """
    ## santity check
    assert isinstance(root, str), TypeError
    assert os.path.exists(root) and os.path.isdir(root), NotADirectoryError
    assert isinstance(lbls, dict), TypeError
    assert ext in __SUPPORTED_MODALITIES__[mod], NotImplementedError

    ## parse kwargs
    if "sample_filter" in kwargs:
        sample_filter = kwargs["sample_filter"]
    else:
        sample_filter = None

    ## initializaion
    seq = ext in __SUPPORTED_IMAGES__[mod]
    samples = set()

    ## traverse all categories/classes/labels
    for _label in os.listdir(root):

        ## bypass invalid labels
        if _label not in lbls:
            continue
        
        _cid = lbls[_label]
        _label_path = os.path.join(root, _label)

        ## travese all video files/image sequences
        for _video in os.listdir(_label_path):
            ## bypass invalid video files
            if (not seq) and (ext not in _video):
                warn_str = "Insane dataset: invalid file {} in path {}".\
                    format(_label, _video)
                logger.warning(warn_str)
                continue
            ## bypass invalid image sequences
            if seq and (not os.path.isdir(_video)):
                warn_str = "Insane dataset: sequence folder {} in path {}".\
                    format(_label, _video)
                logger.warning(warn_str)
                continue
            ## assemble relative path
            _rpath = os.path.join(_label, _video)
            ## strip file extension if it is a video file
            _name = _video if seq else strip_extension(_video)
            ## generate Sample object
            _sample = Sample(root=root, rpath=_rpath, name=_name,
                             mod=mod, ext=ext,
                             lbl=_label, cid=_cid
                            )
            ## filter sample
            if sample_filter is not None:
                if sample_filter(_sample):
                    continue
            ## update sample set
            samples.add(_sample)

    info_str = "[collect_samples] get {} samples from a UCF style layout"\
            .format(len(samples))
    logger.info(info_str)

    return samples


def collect_samples_20bn(root, annots, lbls, mod, ext,
                         **kwargs):
    """ Collect samples from a dataset with a 20BN style layout
    You must provide annotations
    """
    ## santity check
    assert isinstance(root, str), TypeError
    assert os.path.exists(root) and os.path.isdir(root), NotADirectoryError
    assert isinstance(annots), TypeError
    assert isinstance(lbls, dict), TypeError
    assert ext in __SUPPORTED_MODALITIES__[mod], NotImplementedError

    ## parse kwargs
    if "sample_filter" in kwargs:
        sample_filter = kwargs["sample_filter"]
    else:
        sample_filter = None

    ## initializaion
    seq = ext in __SUPPORTED_IMAGES__[mod]
    samples = set()

    ## traverse all video files/image sequences
    for _video in os.listdir(root):
        ## bypass invalid video files
        if (not seq) and (ext not in _video):
            warn_str = "Insane dataset: invalid file {} in path {}".\
                format(_label, _video)
            logger.warning(warn_str)
            continue
        ## bypass invalid image sequences
        if seq and (not os.path.isdir(_video)):
            warn_str = "Insane dataset: sequence folder {} in path {}".\
                format(_label, _video)
            logger.warning(warn_str)
            continue
        ## strip file extension if it is a video file
        _name = _video if seq else strip_extension(_video)
        ## get label
        _label = annots[_name]
        ## bypass invalid labels
        if _label not in lbls:
            continue
        ## get cid
        _cid = lbls[_label]
        _sample = Sample(root=root, rpath=_video, name=_name,
                         mod=mod, ext=ext,
                         lbl=_label, cid=_cid)
        ## filter sample
        if sample_filter is not None:
            if sample_filter(_sample):
                continue
        ## update sample set
        samples.add(_sample)

    info_str = "[collect_samples] get {} samples from a 20BN style layout"\
            .format(len(samples))
    logger.info(info_str)

    return samples


def collect_samples(root, layout, lbls, mod, ext,
                    **kwargs):
    """Collect samples according to given conditions
    @param return set:
        a set of Sample objects
    """
    ## sanity check
    assert layout in __SUPPORTED_LAYOUTS__, NotImplementedError
    if layout == "UCF":
        return collect_samples_ucf(root=root, lbls=lbls, mod=mod, ext=ext,
                                   **kwargs)
    if layout == "20BN":
        if "annots" not in kwargs:
            raise Exception("20BN style layout must specify annotations")
        return collect_samples_20bn(root=root, lbls=lbls, mod=mod, ext=ext,
                                    **kwargs)
    raise NotImplementedError



def test():
    import importlib

    dataset = "weizmann"
    metaset = importlib.import_module(
        "datasets.metadata.metasets.{}".format(dataset))

    kwargs = {
        "root" : metaset.JPG_DATA_PATH,
        "layout" : metaset.layout,
        "lbls" : metaset.__LABELS__,
        "mod" : "RGB",
        "ext" : "jpg",
    }


    samples = collect_samples(**kwargs)
    print(len(samples))

if __name__ == "__main__":
    test()
