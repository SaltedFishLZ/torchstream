"""Basic OpenCV video IO tools testing
"""
import os
import time
from torchstream.io.datapoint import DataPoint
from torchstream.io.conversion import vid2vid
import torchstream.io.backends.opencv as backend
from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download/")
DOWNLOAD_SRC_DIR = "tests/io/conversion/"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


def benchmark_loadtime(src_datapoint, dst_datapoint, load_times=100):
    """test load time
    """
    # test load src time
    st = time.time()
    for _ in range(load_times):
        loader = backend.video2ndarray
        varray = loader(src_datapoint.path)
    ed = time.time()
    avg_load_time = (ed - st) / load_times
    print("[src]:")
    print(src_datapoint.path)
    print("avg load time:", avg_load_time)

    # test load dst time
    load_times = 100
    st = time.time()
    for _ in range(load_times):
        loader = backend.video2ndarray
        varray = loader(dst_datapoint.path)
    ed = time.time()
    avg_load_time = (ed - st) / load_times
    print("[dst]:")
    print(dst_datapoint.path)
    print("avg load time:", avg_load_time)


def test_mp42mp4(benchmarking=False, scale=0.5, fps=10):
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

    mp4_name += "_scale{}_fps{}".format(scale, fps)
    DST_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=mp4_name, ext="mp4")

    # convert
    success = vid2vid(SRC_DATAPOINT, DST_DATAPOINT,
                      scale=0.5, fps=fps)
    assert success

    if benchmarking:
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT)


def test_mp42avi(benchmarking=False, scale=0.5, fps=10):
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

    mp4_name += "_scale{}_fps{}".format(scale, fps)
    DST_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=mp4_name, ext="avi")

    # convert
    success = vid2vid(SRC_DATAPOINT, DST_DATAPOINT, scale=0.5)
    assert success

    if benchmarking:
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT)


def test_webm2avi(benchmarking=False, scale=0.5, fps=10):
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

    webm_name += "_scale{}_fps{}".format(scale, fps)
    DST_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=webm_name, ext="avi")

    # convert
    success = vid2vid(SRC_DATAPOINT, DST_DATAPOINT, retries=10)
    assert success

    if benchmarking:
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT)


if __name__ == "__main__":
    test_mp42mp4(benchmarking=True)
    test_mp42avi(benchmarking=True)
    test_webm2avi(benchmarking=True)
