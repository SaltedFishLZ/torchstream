# -*- coding: utf-8 -*-
# Video Processing Parts
# Author: Zheng Liang
# 

import os
import sys
import glob
import logging

import cv2
import numpy as np 
import psutil


def vid_to_nparray(vpath):
    '''
    Read video from given file path ${vpath} and return the 4-d np array.
    data layout [frame][height][width][channel]
    '''
    cap = cv2.VideoCapture(vpath)
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
        warning_str = "Cannot open video {}: ".format(vpath)
        vm_dict = psutil.virtual_memory()._asdict()
        # suspection: memory overflow
        if (vm_dict['percent'] > 95):
            warning_str += "memory usage {}".format(vm_dict['percent'])
        elif (not os.path.exists(vpath)):
            warning_str += "file not exists"
        else:
            warning_str += "unknown error"

        logging.warning(warning_str)
        return(None)
    f_c = frame.shape[2]
    
    # try to allocate memory for the frames
    try:
        buf = np.empty((f_n, f_h, f_w, f_c), np.dtype('float32'))
    except MemoryError:
        warning_str = "No memory for np array of video {}".format(vpath)
        logging.warning(warning_str)
        cap.release()
        return None

    # keep reading frames from the video
    buf[cnt] = frame
    cnt += 1
    while ((cnt < f_n) and ret):
        ret, buf[cnt] = cap.read()
        cnt += 1
        
    cap.release()
    return buf


def dump_varray(vid_array, tgt_path):
    '''
    Dump 1 video array (4d np array: [frame][height][width][channel]) to
    ${tgt_path}. ${tgt_path} is the directory for images.
    '''
    # check path exists or not
    if (os.path.exists(tgt_path)):
        pass
    else:
        warning_str = "Target {} in dump_varray not exists".format(tgt_path)
        warning_str += ", makedirs for it"
        logging.warning(warning_str)
        os.makedirs(tgt_path)

    # dump pictures
    f_n = vid_array.shape[0]
    cnt = 0
    for _i in range(f_n):
        img_path = os.path.join(tgt_path, "{}.jpg".format(_i))
        ret = cv2.imwrite(img_path, vid_array[_i])
        if ret:
            cnt += 1
        else:
            break
    
    if (cnt == f_n):
        return(True, cnt)
    else:
        return(False, cnt)



def dump_frames(vid_path, tgt_path):
    '''
    Read 1 video from ${vid_path} and dump frames to ${tgt_path}.
    ${vid_path} includes file name. ${tgt_path} is the directory for images.
    '''
    cap = cv2.VideoCapture(vid_path)
    cnt = 0

    # check path exists or not
    if (os.path.exists(vid_path)):
        pass
    else:
        warning_str = "Source {} in dump_frames not exists.".format(vid_path)
        return(False)
    if (os.path.exists(tgt_path)):
        pass
    else:
        warning_str = "Target {} in dump_frames not exists".format(tgt_path)
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
        cv2.imwrite(img_path, frame)

    return (True, cnt)



def dump_ofarray():
    pass



if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    vid_path = os.path.join(dir_path, "test.avi")
    
    varray = vid_to_nparray(vid_path)
    ret, f_n = dump_varray(varray, dir_path)
    for _i in range(f_n):
        os.remove(os.path.join(dir_path, "{}.jpg".format(_i)))
    # dump_frames(vid_path, dir_path)
