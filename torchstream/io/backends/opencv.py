# -*- coding: utf-8 -*-
# Video / Image IO Utilities
# Author: Zheng Liang
# NOTE
# This module only handle data of a certain modality. To fuse
# different modalities (e.g., RGB + Flow), please do it at the
# dataset level
#
# Term / Naming Conversion:
# ┌─────────────┬───────────────────────────────────────────────┐
# │ Frames(s)   |*Description: Separate images(s) from a certain|
# |             | video. Varibale called 'frame(s)' is usually  |
# |             | used as file path.                            |
# |             |*Type: str                                     |
# ├─────────────┼───────────────────────────────────────────────┤
# | Video       |*Description: single video, varible used as    |
# |             | file path.                                    |
# |             |*Type: str                                     |
# ├─────────────┼───────────────────────────────────────────────┤
# │ Frame Array |*Description: Single frame as an Numpy ndarray,|
# |             | stored as [H][W][C] format.                   |
# |             | may referred to as :                          |
# |             | 'farray', 'f_array', 'frm_array', 'fr_array', |
# |             | 'iarray', 'i_array', 'img_array'              |
# |             |*Type: numpy.ndarray('uint8')                  |
# ├─────────────┼───────────────────────────────────────────────┤
# | Video Array |*Description: Single video as an Numpy ndarray,|
# |             | stored as [T][H][W][C] format.                |
# |             | may referred to as:                           |
# |             | 'varray', 'v_array', 'vid_array'              |
# |             |*Type: numpy.ndarray('uint8')                  |
# └─────────────┴───────────────────────────────────────────────┘
#
#

import os
import logging

import cv2
import numpy as np
import psutil

from . import __config__


# local settings (only in dev)
__FRAME_NUM_ERR_LIMIT__ = 10

__SUPPORTED_VIDEO_INPUTS__ = ["avi", "mp4"]
__SUPPORTED_VIDEO_OUTPUTS__ = ["avi", "mp4"]
__SUPPORTED_FRAME_INPUTS__ = ["jpg"]
__SUPPORTED_FRAME_OUTPUTS__ = ["jpg"]
# here, "GRAY" means single-channel data
# some optical-flow based methods may store flow files in jpg
__SUPPORTED_COLORS__ = ["BGR", "RGB", "GRAY"]

# configuring logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(__config__.LOGGER_LEVEL)

# ------------------------------------------------------------------------- #
#               Auxiliary Functions (Not To Be Exported)                    #
# ------------------------------------------------------------------------- #


def failure_suspection(**kwargs):
    """
    """
    reasons = []
    # memory overflow
    vm_dict = psutil.virtual_memory()._asdict()
    if vm_dict["percent"] > 95:
        _reason = "memory usage {};".format(vm_dict["percent"])
        reasons.append(_reason)
    # video missing
    if "vpath" in kwargs:
        if not os.path.exists(kwargs["vpath"]):
            _reason = "file missing"
            reasons.append(_reason)
    return ';'.join(reasons)


def convert_farray_color(farray, cin, cout, **kwargs):
    """
    Args:
        farray : input frame as a Numpy ndarray
        cin : input frame's color space
        cout : output frame's color space
        return value : output frame as a Numpy ndarray
    """
    if (cin == cout):
        return(farray)
    if (cin, cout) == ("BGR", "GRAY"):
        output = cv2.cvtColor(farray, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    elif (cin, cout) == ("BGR", "RGB"):
        output = cv2.cvtColor(farray, cv2.COLOR_BGR2RGB)
    elif (cin, cout) == ("RGB", "GRAY"):
        output = cv2.cvtColor(farray, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    elif (cin, cout) == ("RGB", "BGR"):
        output = cv2.cvtColor(farray, cv2.COLOR_RGB2BGR)
    elif (cin, cout) == ("GRAY", "BGR"):
        output = farray[:, :, 0]
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    elif (cin, cout) == ("GRAY", "RGB"):
        output = farray[:, :, 0]
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    else:
        assert NotImplementedError

    return(output)


def farray_show(caption, farray, cin="RGB", **kwargs):
    _i = convert_farray_color(farray, cin, "BGR")
    # must convert to uint8, see:
    # https://stackoverflow.com/questions/48331211/\
    # how-to-use-cv2-imshow-correctly-for-the-float-image-returned-by-cv2-distancet
    _i = _i.astype(np.uint8)
    cv2.imshow(caption, _i)


# ------------------------------------------------------------------------- #
#                   Main Functions (To Be Used outside)                     #
# ------------------------------------------------------------------------- #

def video2ndarray(video, cin="BGR", cout="RGB", **kwargs):
    """Read video from given file path and return 1 video array.
    Args:
        video (str): input video file path
        cin (str): input video's color space
        cout (str): output ndarray's color space
        return(ndarray): a Numpy ndarray for the video
    """
    # TODO: currenly only support input BGR video
    assert "BGR" == cin, NotImplementedError("Only supported BGR video")

    if __config__.STRICT:
        if not os.path.exists(video):
            warn_str = "[video2frames] src video {} missing".format(video)
            logger.error(warn_str)
            return False

    # open VideoCapture
    cap = cv2.VideoCapture(video)
    if (not cap.isOpened()):
        warn_str = "[video2ndarray] cannot open video {} \
            via cv2.VideoCapture ".format(video)
        logger.warning(warn_str)
        cap.release()
        return None
    cnt = 0

    # get estimated video shape and other parameters
    f_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # TODO: get video channels more elegantly
    # Zheng Liang, I canot find any OpenCV API to get channels of a video
    # NOTE: OpenCV Warning
    # Although VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT) might be inaccurate,
    # if you cannot read the first frame, it is definetely wrong!
    ret, frame = cap.read()
    if not ret:
        warn_str = "[video2ndarray] cannot read frame {} from video {} \
            via cv2.VideoCapture.read(): ".format(cnt, video)
        warn_str += failure_suspection(vpath=video)
        logger.warning(warn_str)
        return(None)
    f_c = frame.shape[2]
    if (cout == "GRAY"):
        varray_shape = (f_n, f_h, f_w, 1)
    else:
        varray_shape = (f_n, f_h, f_w, f_c)

    info_str = "[video2ndarray] video {} estimated shape: {}".format(
        video, varray_shape)
    logger.info(info_str)

    # try to allocate memory for the frames
    try:
        buf = np.empty(varray_shape, np.dtype("uint8"))
    except MemoryError:
        warn_str = "[video2ndarray] no memory for video array of {}".\
            format(video)
        logger.error(warn_str)
        cap.release()
        return None

    # keep reading frames from the video
    # NOTE:
    # Since OpenCV doesn't give accurate frames via CAP_PROP_FRAME_COUNT,
    # we choose the following strategy: how many frames you can decode/read
    # is the frame number.
    buf[cnt, :, :, :] = convert_farray_color(frame, cin, cout)
    cnt += 1
    while ((cnt < f_n) and ret):
        ret, frame = cap.read()
        if not ret:
            break
        buf[cnt, :, :, :] = convert_farray_color(frame, cin, cout)
        cnt += 1
    cap.release()

    # check frame number
    if f_n < cnt:
        raise NotImplementedError
    if f_n > cnt:
        if (f_n-cnt) > __FRAME_NUM_ERR_LIMIT__:
            warn_str = "[video2ndarray] {} - CAP_PROP_FRAME_COUNT {} frames,"
            warn_str += "Get {} frames."
            warn_str = warn_str.format(video, f_n, cnt)
            logger.error(warn_str)
        buf = buf[:cnt, :, :, :]

    # output status
    info_str = ("[video2ndarray] successful: video {},"
                " actual shape {}".format(video, buf.shape))
    logger.info(info_str)

    return buf


def video2frames(video, dst_path, cin="BGR", cout="BGR", **kwargs):
    """Read 1 video from ${video} and dump to frames in ${dst_path}.
    Args:
        video : the input video file path
        dst_path : destination directory for images
        cin : input video's color space
        cout : output frames' color space
        return : (False, 0) if failed; (True, frame count) if succeeded.
            TODO: format string for frames
    """
    # check sanity
    # TODO: currenly only support input BGR video
    assert ("BGR" == cin), "Only supported BGR video"
    if os.path.exists(video):
        pass
    else:
        warn_str = "[video2frames] src video {} missing".format(video)
        logger.warning(warn_str)
        return (False, 0)
    if os.path.exists(dst_path):
        pass
    else:
        warn_str = "[video2frames] dst path {} missing".format(dst_path)
        warn_str += ", makedirs for it"
        logger.warning(warn_str)
        os.makedirs(dst_path)

    # open VideoCapture
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        warn_str = "[video2frames] cannot open video {} \
            via cv2.VideoCapture ".format(video)
        logger.warning(warn_str)
        cap.release()
        return (False, 0)
    cnt = 0
    ret = True

    # dump frames, don't need to get shape of frames
    f_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while ((cnt < f_n) and ret):
        ret, farray = cap.read()
        if not ret:
            break
        # color space conversion
        farray = convert_farray_color(farray, cin, cout)
        # write image files
        frame = os.path.join(dst_path, "{}.jpg".format(cnt))
        ret = cv2.imwrite(frame, farray)
        assert ret, "Cannot write image file {}".format(frame)
        # dump frame successfully
        cnt += 1

    if __config__.STRICT:
        assert cnt > 0, "Cannot read empty video"

    # check frame number
    if 0 > cnt:
        logger.critical("[video2frames] unknown error")
    if cnt == 0:
        logger.error("[video2frames] empty video {}".format(video))
    if abs(f_n - cnt) > __FRAME_NUM_ERR_LIMIT__:
        err_str = "[video2ndarray] [{}] CAP_PROP_FRAME_COUNT {} frames, \
            Read {} frames".format(video, f_n, cnt)
        logger.error(err_str)

    # output status
    info_str = "[video2frames] successful: dst {}, {} frames".\
        format(dst_path, cnt)
    logger.info(info_str)

    # release & return
    cap.release()
    return (True, cnt)


def ndarray2frames(varray, dst_path, cin="RGB", cout="BGR", **kwargs):
    """
    Dump 1 video array ${varray} to frames in ${dst_path}
    - varray : input numpy.ndarray format video
    - cin : input ndarray's color space
    - cout : output frames' color space
    - dst_path : output directory for dumped frames.
    TODO: format string for file name
    """
    # check sanity
    if (os.path.exists(dst_path)):
        pass
    else:
        warn_str = "[ndarray2frames]: target {} missing".format(dst_path)
        warn_str += ", makedirs for it"
        logger.warning(warn_str)
        os.makedirs(dst_path)
    assert not ((varray.shape[3] != 1) and (cin == "GRAY")), \
        "Video array is not a grayscale one, mismatch."
    assert not ((varray.shape[3] != 3) and (cin in ["RGB", "BGR"])), \
        "Video array is not a colorful one, mismatch"

    # dump pictures
    f_n = varray.shape[0]
    cnt = 0
    for _i in range(f_n):
        frame = os.path.join(dst_path, "{}.jpg".format(_i))
        farray = varray[_i, :, :, :]
        farray = convert_farray_color(farray, cin, cout)
        ret = cv2.imwrite(frame, farray)
        assert ret, "Cannot write image file {}".format(frame)
        cnt += 1

    # output status
    info_str = "[ndarray2frames] successful, dst {}".format(dst_path)
    info_str += ", shape {}".format(varray.shape)
    logger.info(info_str)

    return(True, cnt)


def ndarray2video(varray, dst_path, cin="RGB", cout="BGR", fps=12, **kwargs):
    """Write a 4d array to a video file
    """
    t, h, w, c = varray.shape
    assert c == 3, NotImplementedError("Only accept color video now")

    writer = cv2.VideoWriter(dst_path,
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             fps, (w, h))

    success = True
    for i in range(t):
        farray = varray[i, :, :, :]
        farray = convert_farray_color(farray, cin, cout)
        writer.write(farray)
    writer.release()

    # always successful
    return success


def frame2ndarray(frame, cin="BGR", cout="RGB"):
    """Read 1 frame and get a farray
    Args:
        frame : input frame's file path
        cin : input frame's color space
        cout : output ndarray's color space
        return value : the corresponding farray of the frame
    """
    # sanity check
    # TODO
    # read image
    farray = cv2.imread(frame)
    assert farray is not None, "cv2.imread {} failed".format(frame)
    # convert color
    farray = convert_farray_color(farray, cin, cout)

    # output status
    info_str = "[frame2ndarray] successful: reads image {},".format(frame)
    info_str += "shape " + str(farray.shape)
    logger.info(info_str)

    # convert data type
    farray = farray.astype(np.dtype("uint8"))

    return(farray)


def frames2ndarray(frames, cin="BGR", cout="RGB", **kwargs):
    """Read all frames, take them as a continuous video, and get a varray
    Args:
        frames : input frames' file paths
        cin : input video's color space
        cout : output ndarray's color space
        return value : the corresponding varray of all the frames
    """
    # get video shape & check sanity
    _f = len(frames)
    if __config__.STRICT:
        assert _f > 0, "Cannot accept empty video"
    else:
        if (0 == _f):
            warn_str = "[frames2ndarray] empty list, no frames"
            logger.error(warn_str)
    img = frame2ndarray(frames[0], cin, cout)
    _h = img.shape[0]
    _w = img.shape[1]
    _c = img.shape[2]
    varray_shape = (_f, _h, _w, _c)
    buff = np.empty(varray_shape, np.dtype("uint8"))

    # reading frames to ndarray
    buff[0, :, :, :] = img
    cnt = 1
    while (cnt < _f):
        buff[cnt, :, :, :] = frame2ndarray(frames[cnt], cin, cout)
        cnt += 1

    # output status
    info_str = "[frames2ndarray] successful:{} frames read".format(cnt)
    logger.info(info_str)

    return(buff)


# ------------------------------------------------------------------------- #
#              Self-test Utilities (Not To Be Used outside)                 #
# ------------------------------------------------------------------------- #

def test_functions(test_configuration):
    """
    Basic OpenCV video IO tools testing
    """
    vpath = os.path.join(DIR_PATH, "test.avi")

    # read video to varray
    varray = video2ndarray(vpath,
            cin=test_configuration['video_color'],
            cout=test_configuration['varray_color'])
    print(varray.shape)

    # dump video to frames
    ret, f_n = video2frames(vpath,
            os.path.join(DIR_PATH, "test_video2frames"), 
            cin=test_configuration['video_color'],
            cout=test_configuration['frames_color'])
    print('Dumping frames from video finished, {} frames'.format(f_n))

    # dump varray to frames
    ret, f_n = ndarray2frames(varray,
            os.path.join(DIR_PATH, "test_ndarray2frames"), 
            cin=test_configuration['varray_color'],
            cout=test_configuration['frames_color'])
    print('Dumping frames from varray finished, {} frames'.format(f_n))

    # video array to video
    ndarray2video(varray, 
        os.path.join(DIR_PATH, "test_ndarray2video.avi")
        )


def test_cleanup():
    """
    clean up function for all testing
    """
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    import shutil
    shutil.rmtree(os.path.join(DIR_PATH, "test_video2frames"))
    shutil.rmtree(os.path.join(DIR_PATH, "test_array2frames"))


def test():
    test_configuration = {
        "video_color"   : "BGR",
        "varray_color"  : "RGB",
        "frames_color"  : "BGR",
        "imgseq_color"  : "RGB"}
    test_functions(test_configuration)


if __name__ == "__main__":
    test()