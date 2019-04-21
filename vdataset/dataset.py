
import os
import sys
import copy

from sklearn.model_selection import train_test_split

import torch

# supported input data modality and corresponding file types
__supported_modalities__ = ['RGB']
__supported_modality_files__ = {
    'RGB': ['jpg', 'avi', 'mp4']
    }

# ---------------------------------------------------------------- #
#           Dataset Structure Style and Supporting
# ---------------------------------------------------------------- #
# 
# * 1. ucf101 style:
#   Your video dataset must have the following file orgnization:
#   Data Path
#   ├── Class 0
#   │   ├── Video 0
#   |   ├── Video 1
#   |   ├── ...
#   |   └── Video N_0
#   ...
#   |
#   └── Class K ...
#   If you use split image frames rather than an entire video, 
#   {Video i} shall be a folder contain all frames in order.
#   for example:
#   ├── Class 0
#   │   ├── Video 0
#   |   |   ├── 0.jpg
#   |   |   ├── 1.jpg
#   ...
#   Or you can storage video files like video_0.mp4
#   These should be specified via [use_imgs]
#
# * 2. sth-sth style:
__supported_dataset_stucture__ = ['ucf101']



class DatasetStructureParser(object):
    '''

    '''
    def __init__(self, path, style):
        self.path = path
        self.style = style

    def __get_samples__(self):
        '''
        get a list of (videos, annotation)
        '''
        pass



class VideoDataset(torch.utils.data.Dataset):
    '''

    '''
    def __init__(self,
        data_path,
        modalities = {'RGB': 'jpg'},
        use_imgs = True,
        ):
        # data_path santity check
        assert os.path.exists(data_path), "Dataset path not exists"
        self.data_path = copy.deepcopy(data_path)
        # modality santity check
        for _mod in modalities:
            assert _mod in __supported_modalities__, 'Unsupported Modality'
            assert modalities[_mod] in __supported_modality_files__[_mod], \
                ("Unspported input file type for modality: {}".format(_mod))
        self.modalities = copy.deepcopy(modalities)
        # NOTE: currently, video is specified for RGB data (although some flow
        # visualization solutions use video for flow too)
        self.use_imgs = True


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass



class VideoClip(object):
    pass