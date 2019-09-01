import os
from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download")
FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

src = os.path.join(DOWNLOAD_SERVER_PREFIX, "tests/utils/download/hello.log")
dst = os.path.join(DIR_PATH, "hello.log")
download(src, dst)
