__test__    =   True
__strict__  =   True
__verbose__ =   True
__vverbose__=   True

import os
import sys
import copy
import logging
import importlib

# import torch

from . import video
from . import UCF101, HMDB51, Weizmann

# supported input data modality and corresponding file types
__supported_modalities__ = ['RGB']
__supported_modality_files__ = {
    'RGB': ['jpg', 'avi', 'mp4']
    }
__supported_video_files__ = {
    'RGB' : ['avi', 'mp4']
}

# ---------------------------------------------------------------- #
#           Dataset Structure Style and Supporting
# ---------------------------------------------------------------- #
# NOTE: You shall not store other files in the dataset !!! 
#
# * 1. UCF101 style:
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
#   This style applies to the following datasets:
#   * UCF101
#   * HMDB51
#   * Weizmann
#   
# * 2. Kinetics style:
#   Kinetics Dataset already split training, validation, testing into
#   different folders ["train", "val", "test"]. Currently, the test set
#   has no annotations. While training set and validation set each follows
#   the UCF101 style
__supported_dataset_styles__ = ['UCF101']
# key: dataset name, value: structure styles
__supported_datasets__ = {
    # UCF101 styled datasets
    'UCF101':'UCF101', 'HMDB51':'UCF101', 'Weizmann':'UCF101',
    }


label_maps = {
    'UCF101' : UCF101.label_map,
    'HMDB51' : HMDB51.label_map,
    'Weizmann': Weizmann.label_map,
}

class Sample(object):
    '''
    An video sample struct containing the meta-data of a video sample
    '''
    def __init__(self, path, name, ext, lbl="default", cid=-1):
        '''
        path: compelete sample path
        name: file name (without any extension and path)
        ext:  file extension (e.g., avi, mp4), '.' excluded, if it == None
              or "", it means the sample is sliced into several images (not
              limited to RGB modality) and the images are stored in the folder
              which is the "path" mentioned above.
        lbl:  label of the sample, is a unique string in certain dataset
        cid:  class id of the sample, is the numerical representation of label
        '''
        self.path   =   copy.deepcopy(path)
        self.name   =   copy.deepcopy(name)
        self.ext    =   copy.deepcopy(ext)
        self.lbl    =   copy.deepcopy(lbl)
        self.cid    =   copy.deepcopy(cid)

    def __repr__(self):
        string = str(self.name) + '\n'
        string += "[label] : {}  \t".format(self.lbl)
        string += "[cid] : {} \t".format(self.cid)
        string += "[path] : {}".format(self.path)
        return(string)

class VideoCollector(object):
    '''
    A helper class which maintains all sample's meta-data of a certain dataset
    We only deal with Meta-data in it
    TODO: support multiple data format and multiple input modality
    '''
    def __init__(self, root, style, label_map, 
                mod = "RGB", ext= "avi", 
                seek_file=True, split=None):
        '''
        '''
        # santity check
        assert (style in __supported_dataset_styles__), \
            "Unsupported Dataset Struture Style"
        self.root = copy.deepcopy(root)
        self.style = copy.deepcopy(style)
        self.label_map = copy.deepcopy(label_map)
        self.mod = copy.deepcopy(mod)
        self.ext = copy.deepcopy(ext)

        self.samples = []
        if (True == seek_file):
            # NOTE: here we use list.extend !!!
            self.samples.extend(
                self.collect_samples(self.root, self.style, self.label_map,
                    self.mod, self.ext))

    @staticmethod
    def collect_samples(root, style, label_map, mod, ext):
        '''
        collect a list of samples = list of samples, while each sample
        = (video, relative path, class id, label)
        here, video may refers to a collection of images
        '''
        if ("UCF101" == style):
            # get all labels
            labels = []
            for _label in os.listdir(root):
                if (os.path.isdir(os.path.join(root, _label))):
                    labels.append(_label)
            # TODO: check whether we need to sort it
            labels = sorted(labels)
             
            # get all videos' file paths and annotations
            # NOTE: each sample is a tuple = 
            # (vid_path, relative_addr, class_id, label)
            samples = []
            for _label in labels:
                for _video in os.listdir(os.path.join(root, _label)):
                    _path = os.path.join(root, _label, _video)
                    _sample = Sample(_path, _video, ext, 
                                  _label, label_map[_label])
                    samples.append(copy.deepcopy(_sample))
            # return results
            return(samples)               
        else:
            assert True, "Unsupported Dataset Struture Style"        

    def __get_samples__(self):
        return(self.samples)









# class VideoDataset(torch.utils.data.Dataset):
#     '''
#     This shall be an abstract base class. It should never be used in deployment.
#     '''
#     def __init__(self,
#         data_path,
#         modalities = {'RGB': 'jpg'},
#         use_imgs = True,
#         ):
#         # data_path santity check
#         assert os.path.exists(data_path), "Dataset path not exists"
#         self.data_path = copy.deepcopy(data_path)
#         # modality santity check
#         for _mod in modalities:
#             assert _mod in __supported_modalities__, 'Unsupported Modality'
#             assert modalities[_mod] in __supported_modality_files__[_mod], \
#                 ("Unspported input file type for modality: {}".format(_mod))
#         self.modalities = copy.deepcopy(modalities)
#         # NOTE: currently, video is specified for RGB data (although some flow
#         # visualization solutions use video for flow too)
#         self.use_imgs = True

#     def __len__(self):
#         pass

#     def __getitem__(self, idx):
#         assert True, "VideoDataset is abstract, __getitem__ must be overrided"



if __name__ == "__main__":
    collector = VideoCollector(
        Weizmann.raw_data_path,
        __supported_datasets__['Weizmann'],
        label_maps['Weizmann']
        )
    
    for _sample in collector.__get_samples__():
        print(_sample)    
