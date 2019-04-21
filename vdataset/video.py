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




def video2ndarray(vid_path, gray_in=False, gray_out=False):
    '''
    Read video from given file path ${vid_path} and return the 4-d np array.
    data layout [frame][height][width][channel]
    '''
    cap = cv2.VideoCapture(vid_path)
    cnt = 0

    # get video shape and other parameters
    f_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # TODO: get video channels more elegantly
    # Zheng Liang, I canot find any OpenCV API to get channels of a video
    ret, frame = cap.read()
    if (False == ret):
        # assemble warning string
        warning_str = "Cannot open video {}: ".format(vid_path)
        vm_dict = psutil.virtual_memory()._asdict()
        # suspection: memory overflow
        if (vm_dict['percent'] > 95):
            warning_str += "memory usage {}".format(vm_dict['percent'])
        elif (not os.path.exists(vid_path)):
            warning_str += "file not exists"
        else:
            warning_str += "unknown error"

        logging.warning(warning_str)
        return(None)
    # TODO: currenly don't support input grayscale video
    assert (not gray_in), "Grayscale/single-channel input video not supported"
    f_c = frame.shape[2]
    if (gray_out):
        varray_shape = (f_n, f_h, f_w, 1)
    else:
        varray_shape = (f_n, f_h, f_w, f_c)

    if (True == __vverbose__):
        print('video {} shape:'.format(vid_path) + str(varray_shape))

    # try to allocate memory for the frames
    try:
        buf = np.empty(varray_shape, np.dtype('float32'))
    except MemoryError:
        warning_str = "No memory for np array of video {}".format(vid_path)
        logging.warning(warning_str)
        cap.release()
        return None

    # keep reading frames from the video
    if (gray_out):
        buf[cnt] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
    else:
        buf[cnt] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cnt += 1
    while ((cnt < f_n) and ret):
        ret, frame = cap.read()
        if (gray_out):
            buf[cnt] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
        else:
            buf[cnt] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        cnt += 1
        
    cap.release()
    if (True == __verbose__):
        logging.info('video2ndarray finished: video {}'.format(vid_path))
    return buf

def video2frames(vid_path, tgt_path):
    '''
    Read 1 video from ${vid_path} and dump frames to ${tgt_path}.
    ${vid_path} includes file name. ${tgt_path} is the directory for images.
    TODO: format string for frames
    '''
    cap = cv2.VideoCapture(vid_path)
    cnt = 0
    # check santity
    if (os.path.exists(vid_path)):
        pass
    else:
        warning_str = "Source {} in video2frames not exists.".format(vid_path)
        return(False)
    if (os.path.exists(tgt_path)):
        pass
    else:
        warning_str = "Target {} in video2frames not exists".format(tgt_path)
        warning_str += ", makedirs for it"
        logging.warning(warning_str)
        os.makedirs(tgt_path)
    # dump frames
    f_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _i in range(f_n):
        ret, frame = cap.read()
        if (False == ret):
            warning_str = "Read video {} failed".format(vid_path)
            logging.warning(warning_str)
            return(False, cnt)
        cnt += 1
        img_path = os.path.join(tgt_path, "{}.jpg".format(_i))
        ret = cv2.imwrite(img_path, frame)
        assert ret, "Cannot dump frames"
    if (True == __verbose__):
        logging.info('video2frames: successful: tgt {}'.format(tgt_path))
    if (__vverbose__):
        print("video2frames: dumps {} frames".format(cnt))
    
    return (True, cnt)


def ndarray2frames(vid_array, tgt_path, gray_in=False, gray_out=False):
    '''
    Dump 1 video array (4d np array: [frame][height][width][channel]) to
    ${tgt_path}. ${tgt_path} is the directory for images.
    TODO: format string for file name
    '''
    # check santity
    if (os.path.exists(tgt_path)):
        pass
    else:
        warning_str = "ndarray2frames: target {} not exists".format(tgt_path)
        warning_str += ", makedirs for it"
        logging.warning(warning_str)
        os.makedirs(tgt_path)
    assert (not ((vid_array.shape[3] != 1) and (True == gray_in))), \
        "Video array is not a grayscale one."
    assert (not ((vid_array.shape[3] != 3) and (False == gray_in))), \
        "Video array is not a RGB one."
    if (__vverbose__):
        print("vid_array shape: " + str(vid_array.shape))
    # dump pictures
    f_n = vid_array.shape[0]
    cnt = 0
    for _i in range(f_n):
        img_path = os.path.join(tgt_path, "{}.jpg".format(_i))
        if (True == gray_in):
            image = vid_array[_i]
            image.resize(image.shape[0], image.shape[1])
            if (False == gray_out):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                pass
        else:
            if (False == gray_out):
                image = cv2.cvtColor(vid_array[_i], cv2.COLOR_RGB2BGR)
            else:
                image = cv2.cvtColor(vid_array[_i], cv2.COLOR_RGB2GRAY)
        ret = cv2.imwrite(img_path, image)
        if ret:
            cnt += 1
        else:
            break
    # check status
    if (cnt == f_n):
        if (True == __verbose__):
            logging.info('ndarray2frames: successful, tgt {}'.format(tgt_path))
        if (__vverbose__):
            print("ndarray2frames dumps {} frames".format(cnt))        
        return(True, cnt)
    else:
        if (True == __verbose__):
            logging.warning('ndarray2frames: failed, tgt {}'.format(tgt_path))
        if (__vverbose__):
            print("ndarray2frames dumps {} frames".format(cnt))                
        return(False, cnt)


def frame2ndarray(frame, gray_in=False, gray_out=False):
    '''
    frame:      the file name of the image file;
    gray_in:    the input image is gray/single-channel or not;
    gray_out:   the output image is gray/single-channel or not;
    return:     a Numpy ndarray, data layout = [height][weight][channel];
    '''
    # santity check, TODO
    assert (gray_in == gray_out), "Color conversion not supported"
    # read image
    if (gray_in):
        img = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
        img = img[:, :, np.newaxis]
    else:
        img = cv2.imread(frame, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # output status
    if (__verbose__):
        info_str = "frame2ndarray: reads image {} successful.".format(frame)
        info_str += "Shape = " + str(img.shape)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)
    # convert data type
    return(img.astype(np.dtype('float32')))
    
def frames2ndarray(frames, gray_in=False, gray_out=False):
    '''
    frames: a list of image file paths
    return: a Numpy ndarray, data layout = [frame][height][weight][channel]
    '''
    # get video shape & check santity
    _f = len(frames)
    assert _f > 0, "Cannot accept empty video"
    img = frame2ndarray(frames[0], gray_in, gray_out)
    _h = img.shape[0]
    _w = img.shape[1]
    _c = img.shape[2]
    varray_shape = (_f, _h, _w, _c)
    buff = np.empty(varray_shape, np.dtype('float32'))
    # reading frames to ndarray
    buff[0, :, :, :] = img
    cnt = 1
    while (cnt < _f):
        buff[cnt, :, :, :] = frame2ndarray(frames[cnt], gray_in, gray_out)
        cnt += 1
    # output status
    if (__verbose__):
        info_str = "frames2ndarray: {} frames read successful".format(cnt)
        logging.info(info_str)
        if (__vverbose__):
            print(info_str)
    return(buff)


def imshow_float(caption, image, gray_in = False):
    if (gray_in):
        _i = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2BGR)
    else:
        _i = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
    def __init__(self, vid_path, file_type = 'jpg',
            gray_in=False, gray_out=False,
            seek_file=True):
        '''
        TODO: format string for file name
        '''
        self.vid_path = (vid_path)
        self.file_type = (file_type)
        self.gray_in = gray_in
        self.gray_out = gray_out
        
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
        array = frame2ndarray(_fpath, self.gray_in, self.gray_out)
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
        array = frames2ndarray(_fpaths, self.gray_in, self.gray_out)
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
            gray_in=False, gray_out=False):
        
        super(SegmentedImageSequence, self).__init__(
            vid_path, file_type, gray_in, gray_out
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
        gray_in = copy.deepcopy(self.gray_in)
        gray_out = copy.deepcopy(self.gray_out)
        # NOTE: here we replace file_count with seg_num
        file_count = copy.deepcopy(self.seg_num)
        file_list = copy.deepcopy(self.snipt_list)

        _img_seq = ImageSequence(vid_path, file_type,
                gray_in=gray_in, gray_out=gray_out,
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
            gray_in=False, gray_out=False):

        super(ClippedImageSequence, self).__init__(
            vid_path, file_type, gray_in, gray_out
        )   

    def __ImageSequence__(self):
        '''
        Casting function
        '''
        vid_path = copy.deepcopy(self.vid_path)
        file_type = copy.deepcopy(self.file_type)
        gray_in = copy.deepcopy(self.gray_in)
        gray_out = copy.deepcopy(self.gray_out)        
        # TODO
        pass






if __name__ == "__main__":
    
    if (__test__):

        test_components = {
            'basic':True,
            '__get_frame__':False,
            '__get_frames__':True
        }
        test_configuration = {
            'video_gray' : False,
            'varray_gray' : True,
            'frames_gray' : False,
            'imgseq_gray' : False
        }

        if (test_components['basic']):
            # ------------- #
            #  Basic Test   #
            # ------------- # 
            dir_path = os.path.dirname(os.path.realpath(__file__))
            vid_path = os.path.join(dir_path, "test.mp4")
            varray = video2ndarray(vid_path,
                    gray_in=test_configuration['video_gray'],
                    gray_out=test_configuration['varray_gray'])

            # dump varray to frames
            ret, f_n = ndarray2frames(varray, dir_path, 
                    gray_in=test_configuration['varray_gray'],
                    gray_out=test_configuration['frames_gray'])
            print('Dumping frames finished, {} frames'.format(f_n))
            
            # ----------------- #
            #   Advanced Test   #
            # ----------------- # 
            _seq = ImageSequence(dir_path,
                    gray_in=test_configuration['frames_gray'],
                    gray_out=test_configuration['imgseq_gray']
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
                    gray_in=test_configuration['imgseq_gray'])       

                frame_1 = frames[1, :, :, :]
                imshow_float('{}'.format(_f[1]), frame_1,
                    gray_in=test_configuration['imgseq_gray'])    

                frame_2 = frames[2, :, :, :]
                imshow_float('{}'.format(_f[2]), frame_2,
                    gray_in=test_configuration['imgseq_gray'])

                frame_3 = frames[3, :, :, :]
                imshow_float('{}'.format(_f[3]), frame_3,
                    gray_in=test_configuration['imgseq_gray'])

                (cv2.waitKey(0) & 0xFF == ord('q'))
                cv2.destroyAllWindows()


            # clean up
            if (__debug__):
                pass
            else:
                for _i in range(f_n):
                    os.remove(os.path.join(dir_path, "{}.jpg".format(_i)))
