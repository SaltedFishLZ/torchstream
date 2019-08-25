import os
import torchstream.io.backends.opencv as backend
from torchstream.utils.download import download

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


def test_ndarray2frames_avi():
    """Basic OpenCV video IO tools testing
    """
    vpath = os.path.join(DIR_PATH, "test.avi")
    if not os.path.exists(vpath):
        avi_src = "a18:/home/eecs/zhen/video-acc/testbench/test.avi"
        download(avi_src, vpath)

    # read video to varray
    varray = backend.video2ndarray(vpath, cin="BGR", cout="RGB")
    # dump varray to frames
    fdir = os.path.join(DIR_PATH, "ndarray2frames.frames.avi.d")
    ret, f_n = backend.ndarray2frames(varray, fdir, cin="RGB", cout="BGR")
    print('Dumping frames from varray finished, {} frames'.format(f_n))


def test_ndarray2frames_mp4():
    """Basic OpenCV video IO tools testing
    """
    vpath = os.path.join(DIR_PATH, "test.mp4")
    if not os.path.exists(vpath):
        mp4_src = "a18:/home/eecs/zhen/video-acc/testbench/test.mp4"
        download(mp4_src, vpath)

    # read video to varray
    varray = backend.video2ndarray(vpath, cin="BGR", cout="RGB")
    # dump varray to frames
    fdir = os.path.join(DIR_PATH, "ndarray2frames.frames.mp4.d")
    ret, f_n = backend.ndarray2frames(varray, fdir, cin="RGB", cout="BGR")
    print('Dumping frames from varray finished, {} frames'.format(f_n))


if __name__ == "__main__":
    test_ndarray2frames_avi()
    test_ndarray2frames_mp4()