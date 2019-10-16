"""Helper functions for collecting metadata from a dataset
root directory
"""
import os
import logging

from torchstream.io.datapoint import DataPoint, UNKNOWN_LABEL
from torchstream.io.__support__ import SUPPORTED_VIDEOS, \
    SUPPORTED_FILES
from . import __config__

# configuring logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(__config__.LOGGER_LEVEL)


def collect_flat(root, ext, annotations=None,
                 is_valid_datapoint=None,
                 frame_offset=None, frame_tmpl=None):
    """Collecting datapoints from a flat dataset containing
    all datapoints in one folder.
    Args:
        root: dataset root path
        ext: datapoint file extension
        annotations (dict): key: datapoint name (str), value: label (str)
        is_valid_datapoint (function): validation function
        frame_offset (int)[1]: frame path offset
        frame_tmpl (str)[1]: frame path template
        [1] Only valid for image sequence.
    Return:
        list (DataPoint)
    """
    assert isinstance(root, str), TypeError
    assert os.path.exists(root) and os.path.isdir(root), NotADirectoryError
    assert isinstance(ext, str), TypeError

    assert ext in SUPPORTED_FILES["RGB"], \
        NotImplementedError("Unsupport file type [{}]!".format(ext))

    # Echo Parameters
    echo_str = ("[collect_flat]:\n"
                "root: {}\n"
                "ext: {}\n"
                "is_valid_datapoint: {}\n"
                "frame_offset: {}\n"
                "frame_tmpl: {}").format(
                    root, ext, is_valid_datapoint,
                    frame_offset, frame_tmpl
                )
    logger.info(echo_str)

    datapoints = []
    for sample in os.listdir(root):

        # set datapoint name
        name = sample
        if ext in SUPPORTED_VIDEOS["RGB"]:
            # strip the file extension for video files
            if name.endswith("." + ext):
                name = name[: -len("." + ext)]
            # bypass invalid files
            else:
                logger.warn(("[collect_datapoints]: "
                             "invalid file [{}]").format(name))
                continue
        else:
            # image sequence
            pass

        # set label
        label = UNKNOWN_LABEL
        if annotations is not None:
            if name in annotations:
                label = annotations[name]

        # set other key-word arguments
        kwargs = {}
        if frame_offset is not None:
            kwargs["frame_offset"] = frame_offset
        if frame_tmpl is not None:
            kwargs["frame_tmpl"] = frame_tmpl        

        datapoint = DataPoint(root=root, reldir="", name=name,
                              ext=ext, label=label, **kwargs)

        # bypass invalid datapoint
        if is_valid_datapoint is not None:
            if not is_valid_datapoint(datapoint):
                continue

        datapoints.append(datapoint)

    logger.info(
        "collect_datapoints: [{}] datapoints collected".format(
            len(datapoints)
            )
        )

    return datapoints


def collect_folder(root, ext, is_valid_datapoint=None,
                   frame_offset=None, frame_tmpl=None):
    """Collecting datapoints from a folder dataset with datapoints
    distributed in seperate class folders.
    Args:
        root: dataset root path
        ext: datapoint file extension
        is_valid_datapoint (function): validation function
        frame_offset (int)[1]: frame path offset
        frame_tmpl (str)[1]: frame path template
        [1] Only valid for image sequence.
    Return:
        list (DataPoint)
    """
    assert isinstance(root, str), TypeError
    assert os.path.exists(root) and os.path.isdir(root), NotADirectoryError
    assert isinstance(ext, str), TypeError

    assert ext in SUPPORTED_FILES["RGB"], \
        NotImplementedError("Unsupport file type [{}]!".format(ext))

    # Echo Parameters
    echo_str = ("[collect_folder]:\n"
                "root: {}\n"
                "ext: {}\n"
                "is_valid_datapoint: {}\n"
                "frame_offset: {}\n"
                "frame_tmpl: {}").format(
                    root, ext, is_valid_datapoint,
                    frame_offset, frame_tmpl
                )
    logger.info(echo_str)

    datapoints = []
    for label in os.listdir(root):
        label_path = os.path.join(root, label)

        # bypass files
        if not os.path.isdir(label_path):
            logger.warn(("Unexpected file found in root: \n"
                         "{}").format(label_path))
            continue

        for sample in os.listdir(label_path):

            # set datapoint name
            name = sample
            if ext in SUPPORTED_VIDEOS["RGB"]:
                # strip the file extension for video files
                if name.endswith("." + ext):
                    name = name[: -len("." + ext)]
                # bypass invalid files
                else:
                    logger.warn(("[collect_datapoints]: "
                                 "invalid file [{}]").format(name))
                    continue
            else:
                # image sequence
                pass

            # set other key-word arguments
            kwargs = {}
            if frame_offset is not None:
                kwargs["frame_offset"] = frame_offset
            if frame_tmpl is not None:
                kwargs["frame_tmpl"] = frame_tmpl  

            datapoint = DataPoint(root=root, reldir=label, name=name,
                                  ext=ext, label=label, **kwargs)

            # bypass invalid datapoint
            if is_valid_datapoint is not None:
                if not is_valid_datapoint(datapoint):
                    continue

            datapoints.append(datapoint)

    logger.info(
        "collect_datapoints: [{}] datapoints collected".format(
            len(datapoints)
            )
        )

    return datapoints
