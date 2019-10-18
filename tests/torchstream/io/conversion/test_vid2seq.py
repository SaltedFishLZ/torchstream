"""Basic OpenCV video IO tools testing
"""
import os
import time
import shutil

from torchstream.io.datapoint import DataPoint
from torchstream.io.conversion import vid2seq
from torchstream.io.framesampler import RandomSegmentFrameSampler
import torchstream.io.backends.opencv as backend
from torchstream.transforms import Resize
from torchstream.utils.download import download

DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download/")
DOWNLOAD_SRC_DIR = "tests/io/conversion/"

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


def benchmark_loadtime(src_datapoint,
                       dst_datapoint,
                       frame_sampler=None,
                       load_times=10):
    """test load time
    """
    # test load src time
    loader = backend.video2ndarray
    st = time.time()
    for _ in range(load_times):
        varray = loader(src_datapoint.path)
    ed = time.time()
    avg_load_time = (ed - st) / load_times
    print("[src]:")
    print(src_datapoint.path)
    print("avg load time:", avg_load_time)

    # test load dst time
    loader = backend.frames2ndarray
    framepaths = dst_datapoint.framepaths
    if frame_sampler is not None:
        framepaths = frame_sampler(framepaths)
    st = time.time()
    for _ in range(load_times):
        varray = loader(framepaths)  # noqa F841
    ed = time.time()
    avg_load_time = (ed - st) / load_times
    print("[dst]:")
    print(dst_datapoint.path)
    print("avg load time:", avg_load_time)


def test_mp42jpg(benchmarking=False, transform=None):
    mp4_name = "klwBy72NyFc_000179_000189"
    mp4_path = os.path.join(DIR_PATH, mp4_name + ".mp4")

    # download when missing
    if not os.path.exists(mp4_path):
        mp4_src = os.path.join(DOWNLOAD_SERVER_PREFIX,
                               DOWNLOAD_SRC_DIR,
                               mp4_name + ".mp4")
        download(mp4_src, mp4_path)

    SRC_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=mp4_name, ext="mp4")

    jpg_name = mp4_name
    if not os.path.exists(os.path.join(DIR_PATH, jpg_name)):
        os.makedirs(os.path.join(DIR_PATH, jpg_name))
    else:
        shutil.rmtree(os.path.join(DIR_PATH, jpg_name))
        os.makedirs(os.path.join(DIR_PATH, jpg_name))
    DST_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=jpg_name, ext="jpg")

    # convert
    success = vid2seq(SRC_DATAPOINT, DST_DATAPOINT,
                      transform=transform)
    assert success

    if benchmarking:
        # whole video loading time
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT,
                           load_times=10)
        # subsampled video loading time
        print("2 frames")
        frame_sampler = RandomSegmentFrameSampler(2)
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT,
                           frame_sampler=frame_sampler)
        print("8 frames")
        frame_sampler = RandomSegmentFrameSampler(8)
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT,
                           frame_sampler=frame_sampler)
        print("32 frames")
        frame_sampler = RandomSegmentFrameSampler(32)
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT,
                           frame_sampler=frame_sampler)

    # clean up
    shutil.rmtree(DST_DATAPOINT.path)


def test_mp42bmp(benchmarking=False, transform=None):
    mp4_name = "klwBy72NyFc_000179_000189"
    mp4_path = os.path.join(DIR_PATH, mp4_name + ".mp4")

    # download when missing
    if not os.path.exists(mp4_path):
        mp4_src = os.path.join(DOWNLOAD_SERVER_PREFIX,
                               DOWNLOAD_SRC_DIR,
                               mp4_name + ".mp4")
        download(mp4_src, mp4_path)

    SRC_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=mp4_name, ext="mp4")

    bmp_name = mp4_name
    if not os.path.exists(os.path.join(DIR_PATH, bmp_name)):
        os.makedirs(os.path.join(DIR_PATH, bmp_name))
    else:
        shutil.rmtree(os.path.join(DIR_PATH, bmp_name))
        os.makedirs(os.path.join(DIR_PATH, bmp_name))

    DST_DATAPOINT = DataPoint(root=DIR_PATH, reldir="",
                              name=bmp_name, ext="jpg")

    # convert
    success = vid2seq(SRC_DATAPOINT, DST_DATAPOINT,
                      transform=transform)
    assert success

    if benchmarking:
        # whole video loading time
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT,
                           load_times=10)
        # subsampled video loading time
        print("2 frames")
        frame_sampler = RandomSegmentFrameSampler(2)
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT,
                           frame_sampler=frame_sampler)
        print("8 frames")
        frame_sampler = RandomSegmentFrameSampler(8)
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT,
                           frame_sampler=frame_sampler)
        print("32 frames")
        frame_sampler = RandomSegmentFrameSampler(32)
        benchmark_loadtime(SRC_DATAPOINT, DST_DATAPOINT,
                           frame_sampler=frame_sampler)

    # clean up
    shutil.rmtree(DST_DATAPOINT.path)


if __name__ == "__main__":
    print("*" * 80)
    print("Testing Mp4 -> Jpg...")
    print("*" * 80)
    test_mp42jpg(benchmarking=True)

    print("*" * 80)
    print("Testing Mp4 -> bmp...")
    print("*" * 80)
    test_mp42bmp(benchmarking=True)

    print("*" * 80)
    print("Testing Mp4 -> Jpg with Transform...")
    print("*" * 80)
    transform = Resize(331)
    test_mp42jpg(benchmarking=True, transform=transform)
