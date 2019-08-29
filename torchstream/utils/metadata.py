"""Helper functions for collecting metadata from a dataset
root directory
"""
import os
import pickle
import logging

from torchstream.io.datapoint import DataPoint, UNKNOWN_LABEL
from torchstream.io.__support__ import SUPPORTED_IMAGES, SUPPORTED_VIDEOS
from . import __config__

# configuring logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(__config__.LOGGER_LEVEL)


def collect_datapoints(root, ext="jpg",
                       annotations=None,
                       datapoint_filter=None):
    """Collecting datapoints from a flatten dataset containing
    all datapoints in one folder.
    Args:
        root: dataset root path
        ext: datapoint file extension
        annotations (dict): key: datapoint name (str), value: label (str)
    """
    assert isinstance(root, str), TypeError
    assert os.path.exists(root) and os.path.isdir(root), NotADirectoryError
    assert isinstance(ext, str), TypeError

    SUPPORTED_FILES = SUPPORTED_IMAGES["RGB"] + \
        SUPPORTED_VIDEOS["RGB"]
    assert ext in SUPPORTED_VIDEOS["RGB"], \
        NotImplementedError("Unsupport file type [{}]!".format(ext))

    datapoints = []
    for path in os.listdir(root):

        name = path
        if path.endswith(ext):
            name = name[: -(1 + len(ext))]

        label = UNKNOWN_LABEL
        if annotations is not None:
            label = annots[name]

        datapoint = DataPoint(root=root, reldir="",
                              name=name, ext=ext, label=label)
        datapoints.append(datapoint)

    logger.info(
        "collect_datapoints: [{}] datapoints collected".format(
            len(datapoints)
            )
        )

    return datapoints

def collect_folder():
    pass
