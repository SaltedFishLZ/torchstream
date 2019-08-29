"""Helper functions for collecting metadata from a dataset directory
"""
import os
import pickle
import logging
import hashlib

from . import __config__
from .datapoint import DataPoint, DataPointCounter
from ..utils.filesys import strip_extension, touch_date
from .__support__ import __SUPPORTED_MODALITIES__, SUPPORTED_IMAGES
from .__support__ import __SUPPORTED_LAYOUTS__

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)
CACHE_PATH = os.path.join(DIR_PATH, ".cache")

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

# ---------------------------------------------------------------- #
#                       Utility Functions                          #
# ---------------------------------------------------------------- #

def _is_valid_datapoint(path, mod, ext):
    """Check whether a path is a valid datapoint
    """
    if not isinstance(path, str):
        raise TypeError
    if not isinstance(mod, str):
        raise TypeError
    if ext not in __SUPPORTED_MODALITIES__[mod]:
        raise NotImplementedError
    
    seq = ext in SUPPORTED_IMAGES
    # invalid video files
    if (not seq) and (ext not in path):
        warn_str = "Insane dataset: invalid file {}".format(path)
        logger.warning(warn_str)
        return False
            
    # bypass invalid image sequences
    if seq and (not os.path.isdir(path)):
        warn_str = "Insane dataset: sequence folder {}".format(path)
        logger.warning(warn_str)
        return False

    return True



# ---------------------------------------------------------------- #
#            Collect Samples and Get A Set of Samples              #
# ---------------------------------------------------------------- #

def collect_datapoints_ucf101(root, mod, ext, **kwargs):
    """ Collect datapoints from a dataset with a UCF101 style layout
    """
    assert isinstance(root, str), TypeError
    assert os.path.exists(root) and os.path.isdir(root), NotADirectoryError
    assert isinstance(mod, str), TypeError
    assert ext in __SUPPORTED_MODALITIES__[mod], NotImplementedError

    # initializaion
    seq = ext in SUPPORTED_IMAGES[mod]
    datapoints = []

    # traverse all categories/classes/labels
    for label in os.listdir(root):

        label_path = os.path.join(root, label)

        ## travese all video files/image sequences
        for data in os.listdir(label_path):
            
            rpath = os.path.join(label, data)
            data_path = os.path.join(root, rpath)
            if not _is_valid_datapoint(data_path, mod=mod, ext=ext):
                continue

            name = data if seq else strip_extension(data)
            
            ## generate DataPoint object
            datapoint = DataPoint(root=root, rpath=rpath, name=name,
                                  label=label, mod=mod, ext=ext)

            datapoints.append(datapoint)

    logger.info("{} datapoints from a UCF layout".format(len(datapoints)))

    return datapoints


def collect_datapoints_20bn(root, annots, mod, ext, **kwargs):
    """ Collect datapoints from a dataset with a 20BN style layout
    You must provide annotations
    """
    assert isinstance(root, str), TypeError
    assert os.path.exists(root) and os.path.isdir(root), NotADirectoryError
    assert isinstance(annots, dict), TypeError
    assert isinstance(mod, str), TypeError
    assert ext in __SUPPORTED_MODALITIES__[mod], NotImplementedError

    # initializaion
    seq = ext in SUPPORTED_IMAGES[mod]
    datapoints = []

    # traverse all video files/image sequences
    for data in os.listdir(root):
        data_path = os.path.join(root, data)

        # strip file extension if it is a video file
        name = data if seq else strip_extension(data)
        
        label = annots[name]

        datapoint = DataPoint(root=root, rpath=data, name=name, label=label,
                              mod=mod, ext=ext)

        # update datapoint set
        datapoints.append(datapoint)

    logger.info("get {} datapoints from a 20BN layout".format(len(datapoints)))

    return datapoints


def collect_datapoints(root, layout, mod, ext, datapoint_filter=None, **kwargs):
    """Collect datapoints according to given conditions
    Args
    Return:
        a list of DataPoint objects
    """
    assert layout in __SUPPORTED_LAYOUTS__, NotImplementedError

    # Cache Mechanism
    md5 = hashlib.md5(root.encode('utf-8')).hexdigest()
    dir_path = os.path.dirname(root)
    rel_path = os.path.relpath(root, dir_path)
    cache_path = os.path.join(dir_path, ".cache")
    os.makedirs(cache_path, exist_ok=True)
    cache_file = "{}.{}.{}.all{}.datapoints".format(rel_path, mod, ext, md5)
    cache_file = os.path.join(cache_path, cache_file)
    if (
            os.path.exists(cache_file)
            and os.path.isfile(cache_file)
            and touch_date(cache_file) > touch_date(FILE_PATH)
            and touch_date(cache_file) > touch_date(root)
    ):
        warn_str = "[collect_datapoints] find valid cache {}".\
            format(cache_file)
        logger.warning(warn_str)
        with open(cache_file, "rb") as f:
            allpoints = pickle.load(f)
    else:
        # re-generate all datapoints
        warn_str = "[collect_datapoints] regenerating all data points of {}".\
            format(root)
        logger.warning(warn_str)        
        if layout == "UCF101":
            allpoints = collect_datapoints_ucf101(root=root, mod=mod,
                                                  ext=ext, **kwargs)
        elif layout == "20BN":
            if "annots" not in kwargs:
                raise Exception("20BN style layout must specify annotations")
            allpoints = collect_datapoints_20bn(root=root, mod=mod, ext=ext,
                                                 **kwargs)
        else:
            raise NotImplementedError
        
        # dump to cache file
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(allpoints, f)

    # filter datapoints
    if datapoint_filter is not None:
        datapoints = list(filter(datapoint_filter, allpoints))
    else:
        datapoints = allpoints

    return datapoints


def test(dataset):
    import importlib

    metaset = importlib.import_module(
        "torchstream.datasets.metadata.metasets.{}".format(dataset))

    kwargs = {
        "root" : metaset.JPG_DATA_PATH,
        "layout" : metaset.__layout__,
        "mod" : "RGB",
        "ext" : "jpg",
    }

    if hasattr(metaset, "__ANNOTATIONS__"):
        kwargs["annots"] = metaset.__ANNOTATIONS__

    if hasattr(metaset, "JPG_FILE_TMPL"):
        kwargs["tmpl"] = metaset.JPG_FILE_TMPL
    
    if hasattr(metaset, "JPG_IDX_OFFSET"):
        kwargs["offset"] = metaset.JPG_IDX_OFFSET

    # if hasattr(metaset, "AVI_DATA_PATH"):
    #     kwargs["root"] = metaset.AVI_DATA_PATH
    #     kwargs["ext"] = "avi"

    # kwargs["datapoint_filter"] = lambda x: x.label == "pjump"

    print("Collecting Metadata")
    import time
    st_time = time.time()
    datapoints = collect_datapoints(**kwargs)
    print(len(datapoints))
    ed_time = time.time()
    print("collecting time", ed_time - st_time)

    datapoint_counter = DataPointCounter(datapoints)
    print(datapoint_counter)
    
    print(datapoints)

if __name__ == "__main__":
    import sys
    for _i in range(1, len(sys.argv)):
        test(sys.argv[_i])
