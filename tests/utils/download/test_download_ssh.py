import os
from torchstream.utils.download import download_ssh

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

src = "a18:/home/eecs/zhen/video-acc/testbench/hello.log"
dst = os.path.join(DIR_PATH, "hello.log")
download_ssh(src, dst)