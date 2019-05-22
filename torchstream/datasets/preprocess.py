"""Dataset Preprocess Toolbox
"""
__all__ = [
    "vid2vid", "vid2seq"
]

import os
import logging
import subprocess
try:
    from subprocess import DEVNULL # python 3.x
except ImportError:
    DEVNULL = open(os.devnull, "wb")



from . import __config__
from .metadata.sample import Sample
from .utils.vision import video2frames

# ---------------------------------------------------------------- #
#                  Configuring Python Logger                       #
# ---------------------------------------------------------------- #

if __config__.__VERY_VERBOSE__:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s"
    )
elif __config__.__VERY_VERBOSE__:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s - %(levelname)s - %(message)s"
    )
elif __config__.__VERBOSE__:
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
else:
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
logger = logging.getLogger(__name__)



# ---------------------------------------------------------------- #
#                         Main Functions                           #
# ---------------------------------------------------------------- #

## Video -> Video Conversion
#  using ffmpeg to convert videos
#  @param src_sample Sample: source video's meta-data
#  @param dst_sample Sample: destination video's meta-data
def vid2vid(src_sample, dst_sample, **kwargs):
    """
    """
    ## Santity Check 
    if __config__.__STRICT__:
        ## Check Source Oprand
        assert isinstance(src_sample, Sample), \
            TypeError
        assert not src_sample.seq, \
            "Source sample is not a video"
        ## Check Destination Oprand
        assert isinstance(dst_sample, Sample), \
            TypeError
        assert not dst_sample.seq, \
            "Destination sample is not a video"

    success = False
    fails = 0
    retries = 0
    retries = kwargs.get("retries", 0)

    src_vid = src_sample.path
    dst_vid = dst_sample.path
    
    dst_dir = os.path.dirname(dst_vid)
    os.makedirs(dst_dir, exist_ok=True)

    command = " ".join(["ffmpeg", "-i", src_vid, dst_vid, "-y"])

    while fails <= retries:
        _subp = subprocess.run(command, shell=True, check=False,
                               stdout=DEVNULL, stderr=DEVNULL
                              )
        if _subp.returncode == 0:
            success = True
            break
        else:
            fails += 1
    return success


## Video -> Image Sequence Conversion
#  
#  using .video module to slicing videos
#  @param src_sample Sample: source video's meta-data
#  @param dst_sample Sample: destination video's meta-data
def vid2seq(src_sample, dst_sample, **kwargs):
    """
    """
    ## Santity Check 
    if __config__.__STRICT__:
        ## Check Source Oprand
        assert isinstance(src_sample, Sample), \
            TypeError
        assert not src_sample.seq, \
            "Source sample is not a video"
        ## Check Destination Oprand
        assert isinstance(dst_sample, Sample), \
            TypeError
        assert dst_sample.seq, \
            "Destination sample is not a image sequence"


    success = False
    fails = 0
    retries = kwargs.get("retries", 0)

    src_vid = src_sample.path
    dst_seq = dst_sample.path
    os.makedirs(dst_seq, exist_ok=True)

    while fails <= retries:
        ret = video2frames(src_vid, dst_seq)
        success = ret[0]
        if success:
            break
        else:
            fails += 1
    return success






if __name__ == "__main__":

    # self-testing
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    SRC_VID_SAMPLE = Sample(root=DIR_PATH,
                            rpath="test.webm", name="test", ext="webm")
    DST_VID_SAMPLE = Sample(root=DIR_PATH,
                            rpath="test.avi", name="test", ext="avi")
    print(vid2vid(SRC_VID_SAMPLE, DST_VID_SAMPLE, retries=10))

    DST_SEQ_SAMPLE = Sample(root=DIR_PATH,
                            rpath="test_seq", name="test_seq", ext="jpg")
    print(vid2seq(DST_VID_SAMPLE, DST_SEQ_SAMPLE))
