import os
import torchstream.io.backends.opencv as backend
from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download/")
DOWNLOAD_SRC_DIR = "tests/io/backend/opencv"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


def test_video2frames_avi():
    """Basic OpenCV video IO tools testing
    """
    vpath = os.path.join(DIR_PATH, "test.avi")
    if not os.path.exists(vpath):
        avi_src = os.path.join(DOWNLOAD_SERVER_PREFIX,
                               DOWNLOAD_SRC_DIR,
                               "test.avi")
        download(avi_src, vpath)

    # dump video to frames
    fdir = os.path.join(DIR_PATH, "video2frames.frames.avi.d")
    ret, f_n = backend.video2frames(vpath, fdir, cin="BGR", cout="BGR")
    print('Dumping frames from video finished, {} frames'.format(f_n))


def test_video2frames_mp4():
    """Basic OpenCV video IO tools testing
    """
    vpath = os.path.join(DIR_PATH, "test.mp4")
    if not os.path.exists(vpath):
        mp4_src = os.path.join(DOWNLOAD_SERVER_PREFIX,
                               DOWNLOAD_SRC_DIR,
                               "test.mp4")
        download(mp4_src, vpath)

    # dump video to frames
    fdir = os.path.join(DIR_PATH, "video2frames.frames.mp4.d")
    ret, f_n = backend.video2frames(vpath, fdir, cin="BGR", cout="BGR")
    print('Dumping frames from video finished, {} frames'.format(f_n))


if __name__ == "__main__":
    test_video2frames_avi()
    test_video2frames_mp4()
