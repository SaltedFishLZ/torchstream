import os
import torchstream.io.backends.opencv as backend
from torchstream.utils.download import download

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


def test_video2frames():
    """Basic OpenCV video IO tools testing
    """
    vpath = os.path.join(DIR_PATH, "test.avi")
    if not os.path.exists(vpath):
        avi_src = "a18:/home/eecs/zhen/video-acc/testbench/test.avi"
        download(avi_src, vpath)

    # dump video to frames
    fdir = os.path.join(DIR_PATH, "video2frames.frames.d")
    ret, f_n = backend.video2frames(vpath, fdir, cin="BGR", cout="BGR")
    print('Dumping frames from video finished, {} frames'.format(f_n))


if __name__ == "__main__":
    test_video2frames()