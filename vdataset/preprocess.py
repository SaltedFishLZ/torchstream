import os
import subprocess
try:
    from subprocess import DEVNULL # python 3.x
except ImportError:
    DEVNULL = open(os.devnull, "wb")

from . import metadata
from . import video
from .constant import \
    __test__, __profile__, __strict__, __verbose__, __vverbose__, \
    __supported_video_files__, __supported_image_files__


## Video -> Video Conversion
#  
#  using ffmpeg to convert videos
#  @param src_sample Sample: source video's meta-data
#  @param dst_sample Sample: destination video's meta-data
def vid2vid(src_sample, dst_sample, **kwargs):
    """
    """
    ## Santity Check 
    if __strict__:
        ## Check Source Oprand
        assert isinstance(src_sample, metadata.Sample), \
            TypeError
        assert not src_sample.seq, \
            "Source sample is not a video"
        assert src_sample.ext in __supported_video_files__[src_sample.mod], \
            "Source sample video format unsupported"
        ## Check Destination Oprand
        assert isinstance(dst_sample, metadata.Sample), \
            TypeError
        assert not dst_sample.seq, \
            "Destination sample is not a video"
        assert dst_sample.ext in __supported_video_files__[dst_sample.mod], \
            "Destination sample video format unsupported"

    success = False
    fails = 0
    if "retries" in kwargs:
        retries = kwargs["retries"]
    else:
        retries = 0

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
    if __strict__:
        ## Check Source Oprand
        assert isinstance(src_sample, metadata.Sample), \
            TypeError
        assert not src_sample.seq, \
            "Source sample is not a video"
        assert src_sample.ext in __supported_video_files__[src_sample.mod], \
            "Source sample video format unsupported"
        ## Check Destination Oprand
        assert isinstance(dst_sample, metadata.Sample), \
            TypeError
        assert dst_sample.seq, \
            "Destination sample is not a image sequence"
        assert dst_sample.ext in __supported_image_files__[src_sample.mod], \
            "Destination sample image format unsupported"

    success = False
    fails = 0
    if "retries" in kwargs:
        retries = kwargs["retries"]
    else:
        retries = 0

    src_vid = src_sample.path
    dst_seq = dst_sample.path
    os.makedirs(dst_seq, exist_ok=True)

    while fails <= retries:
        ret = video.video2frames(src_vid, dst_seq)
        success = ret[0]
        if success:
            break
        else:
            fails += 1
    return success

if __name__ == "__main__":

    # self-testing
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    SRC_VID_SAMPLE = metadata.Sample(root=DIR_PATH,
                                     path=os.path.join(DIR_PATH, "test.webm"),
                                     name="test", seq=False, ext="webm"
                                     )
    DST_VID_SAMPLE = metadata.Sample(root=DIR_PATH,
                                     path=os.path.join(DIR_PATH, "test.avi"),
                                     name="test",seq=False, ext="avi")
    print(vid2vid(SRC_VID_SAMPLE, DST_VID_SAMPLE, retries=10))

    DST_SEQ_SAMPLE = metadata.Sample(root=DIR_PATH,
                                     path=os.path.join(DIR_PATH, "test_imgs"),
                                     name="test",seq=True, ext="jpg")
    print(vid2seq(DST_VID_SAMPLE, DST_SEQ_SAMPLE))

