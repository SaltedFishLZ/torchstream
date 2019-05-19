import os
import subprocess
try:
    from subprocess import DEVNULL # python 3.x
except ImportError:
    DEVNULL = open(os.devnull, "wb")

from . import metadata
from . import utilities
from .constant import \
    __test__, __profile__, __strict__, __verbose__, __vverbose__, \
    __supported_modalities__, __supported_modality_files__, \
    __supported_video_files__, __supported_color_space__, \
    __supported_dataset_styles__, __supported_datasets__


## Video -> Video Conversion
#  
#  using ffmpeg to convert videos
#  @param src_sample Sample: source video's meta-data
#  @param dst_sample Sample: destination video's meta-data
def vid2vid(src_sample, dst_sample, **kwargs):
    """
    """

    ## Santity Check
    #  
    #  
    if __strict__:
        ## Check Source Oprand
        assert isinstance(src_sample, metadata.Sample), \
            TypeError
        assert src_sample.ext in __supported_video_files__[src_sample.mod], \
            "Source sample is not a video"
        assert not src_sample.seq, \
            "Source sample is not a video"
        ## Check Destination Oprand
        assert isinstance(dst_sample, metadata.Sample), \
            TypeError
        assert dst_sample.ext in __supported_video_files__[dst_sample.mod], \
            "Destination sample is not a video"
        assert not dst_sample.seq, \
            "Destination sample is not a video"

    success = False
    fails = 0
    if "retries" in kwargs:
        retries = kwargs["retries"]
    else:
        retries = 0

    src_vid = src_sample.path
    tgt_vid = dst_sample.path

    tgt_dir = os.path.dirname(tgt_vid)
    os.makedirs(tgt_dir, exist_ok=True)

    command = ["ffmpeg", "-i", src_vid, tgt_vid, "-y"]
    
    while fails <= retries:
        _subp = subprocess.run(command, shell=True, check=False,
                               stdout=DEVNULL, stderr=DEVNULL,
                               **kwargs
                              )

        if _subp.returncode != 0:
            success = True
            break
        else:
            fails += 1
    
    return success


def vid2img(task):
    pass


if __name__ == "__main__":

    # self-testing
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    SRC_VID_SAMPLE = metadata.Sample(root=DIR_PATH,
                                     path=os.path.join(DIR_PATH, "test.webm"),
                                     name="test", seq=False, ext="avi"
                                     )
    DST_VID_SAMPLE = metadata.Sample(root=DIR_PATH,
                                     path=os.path.join(DIR_PATH, "test.mp4"),
                                     name="test",seq=False, ext="mp4")
    print(vid2vid(SRC_VID_SAMPLE, DST_VID_SAMPLE))
