# -*- coding: utf-8 -*-
# Video Processing Parts
# Author: Zheng Liang
# 
# Term:
# frame(s) refers to split image file(s) from a certain video, and used as
# file path in this module.
# ndarray is Numpy ndarray
# video is a video file path

__test__    =   True
__verbose__ =   True
__vverbose__=   True

import os
import sys
import copy
import glob
import logging

import cv2
import numpy as np 
import psutil


from __init__ import __supported_color_space__

# local setting
_frame_num_err_limit_ = 5

def failure_suspection(vid_path, operation = "CapRead"):
    # suspections: (0) memory overflow, (1) video not exists (2) unknown
    vm_dict = psutil.virtual_memory()._asdict()
    if (vm_dict["percent"] > 95):
        reason = "memory usage {}".format(vm_dict["percent"])
    elif (not os.path.exists(vid_path)):
        reason = "file not exists"
    else:
        reason = "unknown error"
    return(reason)

def convert_frame_color(frame, color_in, color_out):
    if (color_in == color_out):
        return(frame)
    if (color_in, color_out) == ("BGR", "GRAY"):
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
    elif (color_in, color_out) == ("BGR", "RGB"):
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif (color_in, color_out) == ("RGB", "GRAY"):
        output = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis]
    elif (color_in, color_out) == ("RGB", "BGR"):
        output = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif (color_in, color_out) == ("GRAY", "BGR"):
        output = frame[:,:,0]
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    elif (color_in, color_out) == ("GRAY", "RGB"):
        output = frame[:,:,0]
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)        
    else:
        assert True, "Unsupported color conversion"
    
    return(output)

def video2ndarray(vid_path, color_in="BGR", color_out="RGB"):
    '''
    Read video from given file path ${vid_path} and return the 4-d np array.
    data layout [frame][height][width][channel]
    '''
    # Check santity
    # TODO: currenly only support input BGR video
    assert ("BGR" == color_in), "Grayscale/1-channel input not supported"
    if (os.path.exists(vid_path)):
        pass
    else:
        warn_str = "[video2frames] src video {} missing".format(vid_path)
        logging.warning(warn_str)
        return(False)
        
    # open VideoCapture
    cap = cv2.VideoCapture(vid_path)
    if (not cap.isOpened()):
        warn_str = "[video2ndarray] cannot open video {} \
            via cv2.VideoCapture ".format(vid_path)
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
            via cv2.VideoCapture.read(): ".format(cnt, vid_path)
        warn_str += failure_suspection(vid_path)
        logging.warning(warn_str)
        return(None)
    f_c = frame.shape[2]
    if (color_out == "GRAY"):
        varray_shape = (f_n, f_h, f_w, 1)
    else:
        varray_shape = (f_n, f_h, f_w, f_c)

    if (__verbose__):
        info_str = "[video2ndarray] video {} estimated shape: {}".format(
            vid_path, varray_shape)
        logging.info(info_str)
        if (True == __vverbose__):
            print(info_str)

    # try to allocate memory for the frames
    try:
        buf = np.empty(varray_shape, np.dtype("float32"))
    except MemoryError:
        warn_str = "[video2ndarray] no memory for Numpy.ndarray of \
            video {}".format(vid_path)
        logging.warning(warn_str)
        cap.release()
        return None

    # keep reading frames from the video
    # NOTE: 
    # Since OpenCV doesn't give accurate frames via CAP_PROP_FRAME_COUNT,
    # we choose the following strategy: how many frames you can decode/read
    # is the frame number.
    buf[cnt,:,:,:] = convert_frame_color(frame, color_in, color_out)
    cnt += 1
    while ((cnt < f_n) and ret):
        ret, frame = cap.read()
        if (not ret):
            break
        buf[cnt,:,:,:] = convert_frame_color(frame, color_in, color_out)       
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
            .format(vid_path, buf.shape)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)
    return buf

def video2frames(vid_path, tgt_path, color_in="BGR", color_out="BGR"):
    '''
    Read 1 video from ${vid_path} and dump frames to ${tgt_path}.
    ${vid_path} includes file name. ${tgt_path} is the directory for images.
    TODO: format string for frames
    '''
    # check santity
    # TODO: currenly only support input BGR video
    assert ("BGR" == color_in), "Grayscale/1-channel input not supported"
    if (os.path.exists(vid_path)):
        pass
    else:
        warn_str = "[video2frames] src video {} missing".format(vid_path)
        logging.warning(warn_str)
        return(False)
    if (os.path.exists(tgt_path)):
        pass
    else:
        warn_str = "[video2frames] tgt path {} missing".format(tgt_path)
        warn_str += ", makedirs for it"
        logging.warning(warn_str)
        os.makedirs(tgt_path)

    # open VideoCapture
    cap = cv2.VideoCapture(vid_path)
    if (not cap.isOpened()):
        warn_str = "[video2frames] cannot open video {} \
            via cv2.VideoCapture ".format(vid_path)
        logging.warning(warn_str)
        cap.release()
        return None
    cnt = 0
    ret = True

    # dump frames, don't need to get shape of frames
    f_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while ((cnt < f_n) and ret):
        ret, frame = cap.read()
        if (False == ret):
            break
        # color space conversion
        frame = convert_frame_color(frame, color_in, color_out)
        # write image files
        img_path = os.path.join(tgt_path, "{}.jpg".format(cnt))   
        ret = cv2.imwrite(img_path, frame)
        assert ret, "Cannot write image file {}".format(img_path)
        # dump frame successfully
        cnt += 1

    # assert cnt > 0, "Cannot read empty video"
    if (0 == cnt):
        logging.warn("Empty video {}".format(vid_path))

    # check frame number
    if (abs(f_n - cnt) > _frame_num_err_limit_):
        warn_str = "[video2ndarray] CAP_PROP_FRAME_COUNT {} frames, \
            Read {} frames".format(f_n, cnt)
        logging.warn(warn_str)

    # output status
    if (__verbose__):
        info_str = "[video2frames] successful: tgt {}, {} frames".format(
            tgt_path, cnt)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)
    
    return (True, cnt)


def ndarray2frames(vid_array, tgt_path, color_in="RGB", color_out="BGR"):
    '''
    Dump 1 video array (4d np array: [frame][height][width][channel]) to
    ${tgt_path}. ${tgt_path} is the directory for images.
    TODO: format string for file name
    '''
    # check santity
    if (os.path.exists(tgt_path)):
        pass
    else:
        warn_str = "[ndarray2frames]: target {} missing".format(tgt_path)
        warn_str += ", makedirs for it"
        logging.warning(warn_str)
        os.makedirs(tgt_path)
    assert (not ((vid_array.shape[3] != 1) and (True == color_in))), \
        "Video array is not a grayscale one, mismatch."
    assert (not ((vid_array.shape[3] != 3) and (False == color_in))), \
        "Video array is not a RGB one, mismatch"
    
    # dump pictures
    f_n = vid_array.shape[0]
    cnt = 0
    for _i in range(f_n):
        img_path = os.path.join(tgt_path, "{}.jpg".format(_i))
        frame = vid_array[_i, :, :, :]
        frame = convert_frame_color(frame, color_in, color_out)
        ret = cv2.imwrite(img_path, frame)
        assert ret, "Cannot write image file {}".format(img_path)
        cnt += 1
    
    # output status
    if (True == __verbose__):
        info_str = "[ndarray2frames] successful, tgt {}".format(tgt_path)
        info_str += ", shape {}".format(vid_array.shape)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)        

    return(True, cnt)


def frame2ndarray(frame, color_in="BGR", color_out="RGB"):
    '''
    frame:      the file name of the image file;
    color_in:    the input image is gray/single-channel or not;
    color_out:   the output image is gray/single-channel or not;
    return:     a Numpy ndarray, data layout = [height][weight][channel];
    '''
    # santity check
    # TODO
    # read image
    img = cv2.imread(frame)
    img = convert_frame_color(img, color_in, color_out)
    # output status
    if (__verbose__):
        info_str = "[frame2ndarray] successful: reads image {},".format(frame)
        info_str += "shape " + str(img.shape)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)
    # convert data type
    return(img.astype(np.dtype("float32")))
    
def frames2ndarray(frames, color_in="BGR", color_out="RGB"):
    '''
    frames: a list of image file paths
    return: a Numpy ndarray, data layout = [frame][height][weight][channel]
    '''
    # get video shape & check santity
    _f = len(frames)
    # assert _f > 0, "Cannot accept empty video"
    if (0 == _f):
        logging.warn("Empty frames {}".format(vid_path))
    img = frame2ndarray(frames[0], color_in, color_out)
    _h = img.shape[0]
    _w = img.shape[1]
    _c = img.shape[2]
    varray_shape = (_f, _h, _w, _c)
    buff = np.empty(varray_shape, np.dtype("float32"))
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


def imshow_float(caption, image, color_in = "RGB"):
    _i = convert_frame_color(image, color_in, "BGR")
    # must convert to uint8, see:
    # https://stackoverflow.com/questions/48331211/how-to-use-cv2-imshow-correctly-for-the-float-image-returned-by-cv2-distancet
    _i = _i.astype(np.uint8)
    cv2.imshow(caption, _i)




class ImageSequence(object):
    '''
    This class is used to manage a video folder containing video frames.
    NOTE: each frame image must be named as %d.<ext> (e.g., 233.jpg), not
    use any padding zero in the file name! And this class only stores file
    pointers
    '''
    def __init__(self, vid_path, file_type = "jpg",
            color_in="BGR", color_out="RGB",
            seek_file=True):
        '''
        TODO: format string for file name
        '''
        self.vid_path = copy.deepcopy(vid_path)
        self.file_type = copy.deepcopy(file_type)
        self.color_in = copy.deepcopy(color_in)
        self.color_out = copy.deepcopy(color_out)
        
        self.file_count = 0
        self.file_list = []
        # seek valid file paths and add them in file list
        if (seek_file):
            _fcnt = 0
            _file_path = self._get_file_path(_fcnt)
            while (os.path.exists(_file_path)):
                self.file_list.append(_file_path)
                _fcnt += 1
                _file_path = self._get_file_path(_fcnt)
            self.file_count = _fcnt

    def _get_file_path(self, idx):
        '''
        a helper function to get the path of a certain frame in the sequence
        '''
        _filename = str(idx) + "." + str(self.file_type)
        _file_path = os.path.join(self.vid_path, _filename)
        return(_file_path)

    def __get_frame__(self, idx):
        '''
        return a Numpy ndarray, data layout: [height][weight][channel]
        '''
        # generate path & santity check
        assert (idx < self.file_count), "Image index overflow"
        _fpath = self._get_file_path(idx)
        # call frame2ndarray to get array
        array = frame2ndarray(_fpath, self.color_in, self.color_out)
        # output status
        if (__verbose__):
            info_str = "ImageSequence: __get_frame__ success, "
            info_str += "shape "+str(array.shape)
            if (__vverbose__):
                print(info_str)
        return(array)

    def __get_frames__(self, indices):
        '''
        data layout: [frame][height][weight][channel]
        '''
        # only enable santity check on debug mode
        if (__debug__):
            for idx in indices:
                assert (idx < self.file_count), "Image index overflow"
        # generate file paths
        _fpaths = []
        for idx in indices:
            _fpaths.append(self._get_file_path(idx))
        # call frames2ndarray to get array
        array = frames2ndarray(_fpaths, self.color_in, self.color_out)
        # output status
        if (__verbose__):
            info_str = "ImageSequence: __get_frames__ success, "
            info_str += "shape "+str(array.shape)
            if (__vverbose__):
                print(info_str)
        return(array)


class SegmentedImageSequence(ImageSequence):
    '''
    This class is used to manage segmented video.
    Although you can use data transform to get a segmented video, it has to
    load all frames and select some frames in it. It is not efficient.
    '''
    def __init__(self, vid_path, seg_num,
            file_type = 'jpg',
            color_in="BGR", color_out="RGB"):
        super(SegmentedImageSequence, self).__init__(
            vid_path, file_type, color_in, color_out
        )
        self.seg_num = seg_num
        self.snipt_list = []

    def __ImageSequence__(self):
        '''
        Casting function
        '''
        # TODO
        vid_path = copy.deepcopy(self.vid_path)
        file_type = copy.deepcopy(self.file_type)
        color_in = copy.deepcopy(self.color_in)
        color_out = copy.deepcopy(self.color_out)
        # NOTE: here we replace file_count with seg_num
        file_count = copy.deepcopy(self.seg_num)
        file_list = copy.deepcopy(self.snipt_list)

        _img_seq = ImageSequence(vid_path, file_type,
                color_in=color_in, color_out=color_out,
                seek_file=False)
        _img_seq.file_list = file_list
        _img_seq.file_count = file_count
        # TODO
        return(_img_seq)

class ClippedImageSequence(ImageSequence):
    '''
    This class is used to manage segmented video.
    Although you can use data transform to get a segmented video, it has to
    load all frames and select some frames in it. It is not efficient.
    '''
    def __init__(self, vid_path, clip_len,
            file_type = 'jpg',
            color_in="BGR", color_out="RGB"):

        super(ClippedImageSequence, self).__init__(
            vid_path, file_type, color_in, color_out
        )   

    def __ImageSequence__(self):
        '''
        Casting function
        '''
        vid_path = copy.deepcopy(self.vid_path)
        file_type = copy.deepcopy(self.file_type)
        color_in = copy.deepcopy(self.color_in)
        color_out = copy.deepcopy(self.color_out)        
        # TODO
        pass






if __name__ == "__main__":
    
    if (__test__):

        test_components = {
            'basic':True,
            '__get_frame__':True,
            '__get_frames__':False
        }
        test_configuration = {
            'video_color'   : "BGR",
            'varray_color'  : "GRAY",
            'frames_color'  : "RGB",
            'imgseq_color'  : "RGB"
        }

        if (test_components['basic']):
            # ------------- #
            #  Basic Test   #
            # ------------- # 
            # read video to varray
            dir_path = os.path.dirname(os.path.realpath(__file__))
            vid_path = os.path.join(dir_path, "test.avi")
            varray = video2ndarray(vid_path,
                    color_in=test_configuration['video_color'],
                    color_out=test_configuration['varray_color'])
            print(varray.shape)

            # dump video to frames
            ret, f_n = video2frames(vid_path,
                    os.path.join(dir_path, "test_video2frames"), 
                    color_in=test_configuration['video_color'],
                    color_out=test_configuration['frames_color'])
            print('Dumping frames finished, {} frames'.format(f_n))

            # dump varray to frames
            ret, f_n = ndarray2frames(varray,
                    os.path.join(dir_path, "test_ndarray2frames"), 
                    color_in=test_configuration['varray_color'],
                    color_out=test_configuration['frames_color'])
            print('Dumping frames finished, {} frames'.format(f_n))
            
            # ----------------- #
            #   Advanced Test   #
            # ----------------- # 
            _seq = ImageSequence(
                    os.path.join(dir_path, "test_ndarray2frames"),
                    color_in=test_configuration['frames_color'],
                    color_out=test_configuration['imgseq_color']
                    )

            if (test_components['__get_frame__']):
                # __get_frame__ test
                _f = 0
                frame = _seq.__get_frame__(_f)
                imshow_float('{}'.format(_f), frame)

                _f = _seq.file_count - 1
                frame = _seq.__get_frame__(_f)
                imshow_float('{}'.format(_f), frame)

                (cv2.waitKey(0) & 0xFF == ord('q'))
                cv2.destroyAllWindows()
            
            if (test_components['__get_frames__']):
                # get frames test
                _n = 4
                step = (_seq.file_count - 1) // _n
                _f = [0, step * 1, step * 2, step * 3]
                frames = _seq.__get_frames__(_f)

                frame_0 = frames[0, :, :, :]
                imshow_float('{}'.format(_f[0]), frame_0,
                    color_in=test_configuration['imgseq_color'])       

                frame_1 = frames[1, :, :, :]
                imshow_float('{}'.format(_f[1]), frame_1,
                    color_in=test_configuration['imgseq_color'])    

                frame_2 = frames[2, :, :, :]
                imshow_float('{}'.format(_f[2]), frame_2,
                    color_in=test_configuration['imgseq_color'])

                frame_3 = frames[3, :, :, :]
                imshow_float('{}'.format(_f[3]), frame_3,
                    color_in=test_configuration['imgseq_color'])

                (cv2.waitKey(0) & 0xFF == ord('q'))
                cv2.destroyAllWindows()


            # clean up
            if (__debug__):
                pass
            else:
                for _i in range(f_n):
                    os.remove(os.path.join(
                        dir_path, "test_video2frames", "{}.jpg".format(_i)))
                    os.remove(os.path.join(
                        dir_path, "test_array2frames", "{}.jpg".format(_i)))
