import os
from torchstream.utils.download import download, download_rsync

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download")
DOWNLOAD_SRC_DIR = "tests/utils/download"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

src = os.path.join(DOWNLOAD_SERVER_PREFIX, DOWNLOAD_SRC_DIR, "hello.log")
dst = os.path.join(DIR_PATH, "hello.log")
download(src, dst)

src = os.path.join(DOWNLOAD_SERVER_PREFIX, DOWNLOAD_SRC_DIR, "hello.d/")
dst = os.path.join(DIR_PATH, "hello.d")
download_rsync(src, dst)
