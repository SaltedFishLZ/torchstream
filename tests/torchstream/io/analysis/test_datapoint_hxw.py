import os

from torchstream.utils.download import download
from torchstream.io.datapoint import DataPoint
from torchstream.io.analysis import datapoint_hxw

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download/")
DOWNLOAD_SRC_DIR = "tests/io/conversion/"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


def test_datapoint_hxw():
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

    print(datapoint_hxw(SRC_DATAPOINT))


if __name__ == "__main__":
    test_datapoint_hxw()
