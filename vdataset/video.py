# -*- coding: utf-8 -*-
# Video Processing Parts
# Author: Zheng Liang
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
import sys
import copy
import glob
import math
import shutil
import logging

import cv2
import numpy as np 
import psutil

from .__init__ import __test__, __strict__, __verbose__, __vverbose__, \
    __supported_color_space__

# local settings
_frame_num_err_limit_ = 5

__verbose__ = False
__vverbose__ = False

# ------------------------------------------------------------------------- #
#               Auxiliary Functions (Not To Be Exported)                    #
# ------------------------------------------------------------------------- #

def failure_suspection(vid_path, operation="CapRead"):
    # suspections:
    # - memory overflow
    # - video not exists
    # - unknown
    vm_dict = psutil.virtual_memory()._asdict()
    if (vm_dict["percent"] > 95):
        reason = "memory usage {}".format(vm_dict["percent"])
    elif (not os.path.exists(vid_path)):
        reason = "file not exists"
    else:
        reason = "unknown error"
    return(reason)

def convert_farray_color(farray, color_in, color_out):
    '''
    - farray : input frame as a Numpy ndarray
    - color_in : input frame's color space
    - color_out : output frame's color space
    - return value : output frame as a Numpy ndarray   
    '''
    if (color_in == color_out):
        return(farray)
    if (color_in, color_out) == ("BGR", "GRAY"):
        output = cv2.cvtColor(farray, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
    elif (color_in, color_out) == ("BGR", "RGB"):
        output = cv2.cvtColor(farray, cv2.COLOR_BGR2RGB)
    elif (color_in, color_out) == ("RGB", "GRAY"):
        output = cv2.cvtColor(farray, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
    elif (color_in, color_out) == ("RGB", "BGR"):
        output = cv2.cvtColor(farray, cv2.COLOR_RGB2BGR)
    elif (color_in, color_out) == ("GRAY", "BGR"):
        output = farray[:,:,0]
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    elif (color_in, color_out) == ("GRAY", "RGB"):
        output = farray[:,:,0]
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)        
    else:
        assert True, "Unsupported color conversion"
    
    return(output)

def farray_show(caption, farray, color_in="RGB"):
    _i = convert_farray_color(farray, color_in, "BGR")
    # must convert to uint8, see:
    # https://stackoverflow.com/questions/48331211/\
    # how-to-use-cv2-imshow-correctly-for-the-float-image-returned-by-cv2-distancet
    _i = _i.astype(np.uint8)
    cv2.imshow(caption, _i)



# ------------------------------------------------------------------------- #
#                   Main Functions (To Be Used outside)                     #
# ------------------------------------------------------------------------- #

def video2ndarray(video, color_in="BGR", color_out="RGB"):
    '''
    Read video from given file path ${video} and return the video array.
    - video : input video file path
    - color_in : input video's color space
    - color_out : output ndarray's color space
    - return value : a Numpy ndarray for the video
    '''
    # Check santity
    # TODO: currenly only support input BGR video
    assert ("BGR" == color_in), "Only supported BGR video"
    if (os.path.exists(video)):
        pass
    else:
        warn_str = "[video2frames] src video {} missing".format(video)
        logging.warning(warn_str)
        return(False)
        
    # open VideoCapture
    cap = cv2.VideoCapture(video)
    if (not cap.isOpened()):
        warn_str = "[video2ndarray] cannot open video {} \
            via cv2.VideoCapture ".format(video)
        logging.warning(warn_str)
        cap.release()
        return None
    cnt = 0

    # get video shape and other parameters
    f_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # TODO: get video channels more elegantly
    # Zheng Liang, I canot find any OpenCV API to get channels of a video
    # NOTE: OpenCV Warning
    # Although VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT) might be inaccurate,
    # if you cannot read the first frame, it is definetely wrong!
    ret, frame = cap.read()
    if (False == ret):
        warn_str = "[video2ndarray] cannot read frame {} from video {} \
            via cv2.VideoCapture.read(): ".format(cnt, video)
        warn_str += failure_suspection(video)
        logging.warning(warn_str)
        return(None)
    f_c = frame.shape[2]
    if (color_out == "GRAY"):
        varray_shape = (f_n, f_h, f_w, 1)
    else:
        varray_shape = (f_n, f_h, f_w, f_c)

    if (__verbose__):
        info_str = "[video2ndarray] video {} estimated shape: {}".format(
            video, varray_shape)
        logging.info(info_str)
        if (True == __vverbose__):
            print(info_str)

    # try to allocate memory for the frames
    try:
        buf = np.empty(varray_shape, np.dtype("uint8"))
    except MemoryError:
        warn_str = "[video2ndarray] no memory for video array of \
            {}".format(video)
        logging.warning(warn_str)
        cap.release()
        return None

    # keep reading frames from the video
    # NOTE: 
    # Since OpenCV doesn't give accurate frames via CAP_PROP_FRAME_COUNT,
    # we choose the following strategy: how many frames you can decode/read
    # is the frame number.
    buf[cnt,:,:,:] = convert_farray_color(frame, color_in, color_out)
    cnt += 1
    while ((cnt < f_n) and ret):
        ret, frame = cap.read()
        if (not ret):
            break
        buf[cnt,:,:,:] = convert_farray_color(frame, color_in, color_out)       
        cnt += 1 
    cap.release()

    # check frame number
    if (f_n > cnt):        
        if ((f_n-cnt) > _frame_num_err_limit_):
            warn_str = "[video2ndarray] CAP_PROP_FRAME_COUNT {} frames, \
                Read {} frames".format(f_n, cnt)
            logging.warn(warn_str)
        # slice the buffder
        buf = buf[:cnt,:,:,:]

    # output status
    if (True == __verbose__):
        info_str = "[video2ndarray] successful: video {}, actual shape {}"\
            .format(video, buf.shape)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)
    return buf


def video2frames(video, dst_path, color_in="BGR", color_out="BGR"):
    '''
    Read 1 video from ${video} and dump to frames in ${dst_path}.
    - video : the input video file path
    - dst_path : destination directory for images
    - color_in : input video's color space
    - color_out : output frames' color space
    - return value : (False, 0) if failed; (True, ${frame count}) if \
succeeded.
    TODO: format string for frames
    '''
    # check santity
    # TODO: currenly only support input BGR video
    assert ("BGR" == color_in), "Only supported BGR video"
    if (os.path.exists(video)):
        pass
    else:
        warn_str = "[video2frames] src video {} missing".format(video)
        logging.warning(warn_str)
        return (False, 0)
    if (os.path.exists(dst_path)):
        pass
    else:
        warn_str = "[video2frames] dst path {} missing".format(dst_path)
        warn_str += ", makedirs for it"
        logging.warning(warn_str)
        os.makedirs(dst_path)

    # open VideoCapture
    cap = cv2.VideoCapture(video)
    if (not cap.isOpened()):
        warn_str = "[video2frames] cannot open video {} \
            via cv2.VideoCapture ".format(video)
        logging.warning(warn_str)
        cap.release()
        return (False, 0)
    cnt = 0
    ret = True

    # dump frames, don't need to get shape of frames
    f_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while ((cnt < f_n) and ret):
        ret, farray = cap.read()
        if (False == ret):
            break
        # color space conversion
        farray = convert_farray_color(farray, color_in, color_out)
        # write image files
        frame = os.path.join(dst_path, "{}.jpg".format(cnt))   
        ret = cv2.imwrite(frame, farray)
        assert ret, "Cannot write image file {}".format(frame)
        # dump frame successfully
        cnt += 1
    
    if __strict__:
        assert cnt > 0, "Cannot read empty video"
    else:
        if (0 == cnt):
            logging.warn("[video2frames] empty video {}".format(video))

    # check frame number
    if (abs(f_n - cnt) > _frame_num_err_limit_):
        warn_str = "[video2ndarray] CAP_PROP_FRAME_COUNT {} frames, \
            Read {} frames".format(f_n, cnt)
        logging.warn(warn_str)

    # output status
    if (__verbose__):
        info_str = "[video2frames] successful: dst {}, {} frames".format(
            dst_path, cnt)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)
    
    cap.release()
    return (True, cnt)


def ndarray2frames(varray, dst_path, color_in="RGB", color_out="BGR"):
    '''
    Dump 1 video array ${varray} to frames in ${dst_path}
    - varray : input numpy.ndarray format video
    - color_in : input ndarray's color space
    - color_out : output frames' color space    
    - dst_path : output directory for dumped frames.
    TODO: format string for file name
    '''
    # check santity
    if (os.path.exists(dst_path)):
        pass
    else:
        warn_str = "[ndarray2frames]: target {} missing".format(dst_path)
        warn_str += ", makedirs for it"
        logging.warning(warn_str)
        os.makedirs(dst_path)
    assert (not ((varray.shape[3] != 1) and (True == color_in))), \
        "Video array is not a grayscale one, mismatch."
    assert (not ((varray.shape[3] != 3) and (False == color_in))), \
        "Video array is not a RGB one, mismatch"
    
    # dump pictures
    f_n = varray.shape[0]
    cnt = 0
    for _i in range(f_n):
        frame = os.path.join(dst_path, "{}.jpg".format(_i))
        farray = varray[_i, :, :, :]
        farray = convert_farray_color(farray, color_in, color_out)
        ret = cv2.imwrite(frame, farray)
        assert ret, "Cannot write image file {}".format(img_path)
        cnt += 1
    
    # output status
    if (True == __verbose__):
        info_str = "[ndarray2frames] successful, dst {}".format(dst_path)
        info_str += ", shape {}".format(varray.shape)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)        

    return(True, cnt)


def frame2ndarray(frame, color_in="BGR", color_out="RGB"):
    '''
    Read 1 frame and get a farray
    - frame : input frame's file path
    - color_in : input frame's color space
    - color_out : output ndarray's color space  
    - return value : the corresponding farray of the frame
    '''
    # santity check
    # TODO
    # read image
    farray = cv2.imread(frame)
    farray = convert_farray_color(farray, color_in, color_out)
    # output status
    if (__verbose__):
        info_str = "[frame2ndarray] successful: reads image {},".format(frame)
        info_str += "shape " + str(farray.shape)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)
    # convert data type
    farray = farray.astype(np.dtype("uint8"))
    
    return(farray)


def frames2ndarray(frames, color_in="BGR", color_out="RGB"):
    '''
    Read all frames, take them as a continuous video, and get a varray
    - frames : input frames' file paths
    - color_in : input video's color space
    - color_out : output ndarray's color space  
    - return value : the corresponding varray of all the frames
    '''
    # get video shape & check santity
    _f = len(frames)
    if (__strict__):
        assert _f > 0, "Cannot accept empty video"
    else:
        if (0 == _f):
            warn_str = "[frames2ndarray] empty list, no frames"
            logging.warn(warn_str)
    img = frame2ndarray(frames[0], color_in, color_out)
    _h = img.shape[0]
    _w = img.shape[1]
    _c = img.shape[2]
    varray_shape = (_f, _h, _w, _c)
    buff = np.empty(varray_shape, np.dtype("uint8"))
    # reading frames to ndarray
    buff[0, :, :, :] = img
    cnt = 1
    while (cnt < _f):
        buff[cnt, :, :, :] = frame2ndarray(frames[cnt], color_in, color_out)
        cnt += 1
    # output status
    if (__verbose__):
        info_str = "[frames2ndarray] successful:{} frames read".format(cnt)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)
    return(buff)



# ------------------------------------------------------------------------- #
#                   Main Classes (To Be Used outside)                       #
# ------------------------------------------------------------------------- #

class ImageSequence(object):
    '''
    This class is used to manage a video folder containing video frames.
    The folder shall looks like this:
    video path
    ├── frame 0
    ├── frame 1
    ├── ...
    └── frame N
    NOTE: Each frame must be named as %d.<ext> (e.g., 233.jpg). Do not
    use any padding zero in the file name! And this class only stores file
    pointers
    TODO: support multiple modalities (multiple file extension)

    Stable API: _get_varray_
    '''
    def __init__(self, path, 
            ext = "jpg", color_in="BGR", color_out="RGB"):
        '''
        TODO: format string for file name
        '''
        self.path = copy.deepcopy(path)
        self.ext = copy.deepcopy(ext)
        self.color_in = copy.deepcopy(color_in)
        self.color_out = copy.deepcopy(color_out)
        
        self.fcount = 0     # frame counnt
        self.fids = []      # frame ids

        # seek all valid frames and add their indices
        _fcnt = 0
        _file_path = self._get_frame_path_(_fcnt)
        while (os.path.exists(_file_path)):
            _fcnt += 1         
            _file_path = self._get_frame_path_(_fcnt)
        
        if (__strict__):
            assert (_fcnt > 0), "Empty video folder {}".format(path)
        else:
            if (0 == _fcnt):
                warn_str = "ImageSequence: [__init__] "
                warn_str += "empty video folder {}".format(path)
                logging.warn(warn_str)

        self.fcount = _fcnt
        self.fids = list(range(_fcnt))

    def _get_frame_path_(self, idx):
        '''
        get the path of idx-th frame
        NOTE: currently, we all use the original frame index
        '''
        _filename = str(idx) + "." + str(self.ext)
        _file_path = os.path.join(self.path, _filename)
        return(_file_path)

    def _get_farray_(self, idx):
        '''
        get the farray of the idx-th frame
        '''
        # generate path & santity check
        assert (idx <= self.fids[self.fcount - 1]), \
            "Image index [{}] exceeds max fid[{}]"\
            .format(idx, self.fids[self.fcount - 1])
        _fpath = self._get_frame_path_(idx)
        # call frame2ndarray to get image array
        farray = frame2ndarray(_fpath, self.color_in, self.color_out)
        # output status
        if (__verbose__):
            info_str = "ImageSequence: _get_farray_ success, "
            info_str += "shape "+str(farray.shape)
            logging.info(info_str)
            if (__vverbose__):
                print(info_str)
        return(farray)

    def _get_varray_(self, indices=None):
        '''
        get the varray of all the frames, if indices == None.
        otherwise get certain frames as they are a continuous video
        '''
        # only enable santity check in strict mode for higher perfomance
        if (__strict__):
            if (None == indices):
                indices = self.fids
            else:
                for idx in indices:
                    assert (idx < self.fcount), "Image index {} overflow"\
                        .format(idx)
        # generate file paths
        _fpaths = []
        for idx in indices:
            _fpaths.append(self._get_frame_path_(idx))
        # call frames2ndarray to get array
        varray = frames2ndarray(_fpaths, self.color_in, self.color_out)
        # output status
        if (__verbose__):
            info_str = "ImageSequence: _get_varray_ success, "
            info_str += "shape "+str(varray.shape)
            logging.info(info_str)
            if (__vverbose__):
                print(info_str)
        return(varray)


class ClippedImageSequence(ImageSequence):
    '''
    This class is used to manage clipped video.
    Although you can use data transform to clip an entire video, it has to
    load all frames and select some frames in it. It is not efficient if you
    have preprocessed the video and dump all frames.
    '''
    def __init__(self, path, clip_len, 
            ext = 'jpg', color_in="BGR", color_out="RGB"):
        super(ClippedImageSequence, self).__init__(
            path, ext, color_in, color_out)
        print("DEBUG")
        print(clip_len)
        self.fids = self.__clip__(self.fids, self.fcount, clip_len)
        self.fcount = copy.deepcopy(clip_len)

    @staticmethod
    def __clip__(fids, fcount, clip_len):
        '''
        __clip__ is made an independent function in case you may need to use
        it in other places
        '''
        assert (fcount >= clip_len), \
            "Clip length [{}] exceeds video length [{}]"\
                .format(clip_len, fcount)
        
        if (fcount == clip_len):
            return(fids)
        else:
            # random jitter in time dimension, and re-sample frames
            offset = np.random.randint(fcount - clip_len)
            if (__verbose__):
                info_str = "ClippedImageSequence: [__clip__] "
                info_str += "clip frames [{}, {})".\
                    format(offset, offset + clip_len)
                logging.info(info_str)
                if (__vverbose__):
                    print(info_str)
            return(fids[offset : offset + clip_len])


class SegmentedImageSequence(ImageSequence):
    '''
    This class is used to manage segmented video.
    Although you can use data transform to get a segmented video, it has to
    load all frames and select some frames in it. It is not efficient.
    '''
    def __init__(self, path, seg_num,
            ext = 'jpg', color_in="BGR", color_out="RGB"):
        super(SegmentedImageSequence, self).__init__(
            path, ext, color_in, color_out)    
        self.fids = self.__segment__(self.fids, self.fcount, seg_num)
        self.fcount = copy.deepcopy(seg_num)


    @staticmethod
    def __segment__(frames, fcount, seg_num):
        '''
        __segment__ is made an independent function in case you may need to 
        re-use it in other places       
        '''
        assert (seg_num > 0), "Segment number must > 0"
        assert (fcount >= seg_num), \
            "Segment number [{}] exceeds video length [{}]".\
                format(seg_num, fcount)

        # interval (length of each segment) = ceil(fcount/seg_num)
        # ((a + b - 1) // b) == ceil(a/b)
        _interval = (fcount + seg_num - 1) // seg_num
        _residual = fcount - _interval * (seg_num - 1)
        _kfids = []         # key frame ids

        # original TSN uses random key frames
        if (_residual == 0):
            for _i in range(seg_num):
                _idx = _i * _interval + np.random.randint(_interval)
                _kfids.append(_idx)
        else:
            for _i in range(seg_num - 1):
                _idx = _i * _interval + np.random.randint(_interval)
                _kfids.append(_idx)
            _idx = _interval * (seg_num - 1) + np.random.randint(_residual)
            _kfids.append(_idx)            

        if (__verbose__):
            info_str = "SegmentedImageSequence: [__segment__] "
            info_str += "key frames {}".format(_kfids)
            logging.info(info_str)
            if (__vverbose__):
                print(info_str)

        return(_kfids)




# ------------------------------------------------------------------------- #
#              Self-test Utilities (Not To Be Used outside)                 #
# ------------------------------------------------------------------------- #

def test_functions(test_configuration):
    '''
    Basic OpenCV video IO tools testing
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    vid_path = os.path.join(dir_path, "test.avi")
    # read video to varray
    varray = video2ndarray(vid_path,
            color_in=test_configuration['video_color'],
            color_out=test_configuration['varray_color'])
    print(varray.shape)
    # dump video to frames
    ret, f_n = video2frames(vid_path,
            os.path.join(dir_path, "test_video2frames"), 
            color_in=test_configuration['video_color'],
            color_out=test_configuration['frames_color'])
    print('Dumping frames from video finished, {} frames'.format(f_n))
    # dump varray to frames
    ret, f_n = ndarray2frames(varray,
            os.path.join(dir_path, "test_ndarray2frames"), 
            color_in=test_configuration['varray_color'],
            color_out=test_configuration['frames_color'])
    print('Dumping frames from varray finished, {} frames'.format(f_n))


def test_classes(test_components, test_configuration): 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    _seq = SegmentedImageSequence(
            os.path.join(dir_path, "test_ndarray2frames"),
            seg_num = 16,
            color_in=test_configuration['frames_color'],
            color_out=test_configuration['imgseq_color']
            )

    # _get_farray_ test
    if (test_components['_get_farray_']):
        _i = 0
        frame = _seq._get_farray_(_seq.fids[_i])
        farray_show('{}'.format(_seq.fids[_i]), frame)

        _i = _seq.fcount - 1
        frame = _seq._get_farray_(_seq.fids[_i])
        farray_show('{}'.format(_seq.fids[_i]), frame)

        (cv2.waitKey(0) & 0xFF == ord('q'))
        cv2.destroyAllWindows()
    
    # _get_varray_ test
    if (test_components['_get_varray_']):
        varray = _seq._get_varray_()
        print(varray.shape)
        _f = varray.shape[0]
        for _i in range(_f):
            farray_show("{}".format(_seq.fids[_i]), varray[_i,:,:,:])
            (cv2.waitKey(0) & 0xFF == ord('q'))
            cv2.destroyAllWindows()


def test_cleanup():
    '''
    clean up function for all testing
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    shutil.rmtree(os.path.join(dir_path, "test_video2frames"))
    shutil.rmtree(os.path.join(dir_path, "test_array2frames"))




if __name__ == "__main__":
    
    if (__test__):

        test_components = {
            'functions' : True,
            'classes' : {'_get_farray_':True,
                '_get_varray_':True
                }
        }
        test_configuration = {
            'video_color'   : "BGR",
            'varray_color'  : "RGB",
            'frames_color'  : "BGR",
            'imgseq_color'  : "RGB"
        }

        if (test_components['functions']):

            test_functions(test_configuration)

            test_classes(test_components['classes'], test_configuration)

            # clean up
            if (__debug__):
                pass
            else:
                test_cleanup()