"""Basic OpenCV video IO tools testing
"""
import os
from torchstream.io.datapoint import DataPoint
from torchstream.io.conversion import vid2vid
from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download/")
DOWNLOAD_SRC_DIR = "tests/io/conversion/"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


def test_mp42avi():
    mp4_name = "W5GWm_g9X1s_000095_000105"
    mp4_path = os.path.join(DIR_PATH, mp4_name + ".mp4")

    # download when missing
    if not os.path.exists(mp4_path):
        mp4_src = os.path.join(DOWNLOAD_SERVER_PREFIX,
                               DOWNLOAD_SRC_DIR,
                               mp4_name + ".mp4")
        download(mp4_src, mp4_path)

    SRC_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=mp4_name, ext="mp4")
    DST_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=mp4_name, ext="avi")

    success = vid2vid(SRC_DATAPOINT, DST_DATAPOINT)
    assert success


def test_webm2avi():
    webm_name = "1"
    webm_path = os.path.join(DIR_PATH, webm_name + ".webm")

    # download when missing
    if not os.path.exists(webm_path):
        webm_src = os.path.join(DOWNLOAD_SERVER_PREFIX,
                                DOWNLOAD_SRC_DIR,
                                webm_name + ".webm")
        download(webm_src, webm_path)

    SRC_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=webm_name, ext="webm")
    DST_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=webm_name, ext="avi")

    success = vid2vid(SRC_DATAPOINT, DST_DATAPOINT, retries=10)
    assert success


if __name__ == "__main__":
    test_mp42avi()
    test_webm2avi()
