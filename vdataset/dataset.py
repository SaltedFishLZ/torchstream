__test__    =   True
__strict__  =   True
__verbose__ =   True
__vverbose__=   True

import os
import sys
import copy
import multiprocessing as mp

from sklearn.model_selection import train_test_split

# import torch


import video

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
# * 1. UCF-101 style:
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
__supported_dataset_stucture_styles__ = ['UCF-101']
__supported_dataset__ = ['UCF-101']


class VideoCollector(object):
    '''

    '''
    def __init__(self, path, style, label_map,
        data_mod = "RGB", file_ext= "avi",
        seek_file=True):
        '''
        file_ext is '' or image formats(e.g., 'jpg') means the video is a
        collection of images
        '''
        # santity check
        assert (style in __supported_dataset_stucture_styles__), \
            "Unsupported Dataset Struture Style"
        self.path = copy.deepcopy(path)
        self.style = copy.deepcopy(style)
        self.label_map = copy.deepcopy(label_map)
        self.data_mod = copy.deepcopy(data_mod)
        self.file_ext = copy.deepcopy(file_ext)

        self.samples = []
        if (True == seek_file):
            # NOTE: here we use list.extend !!!
            self.samples.extend(
                self.collect_samples(self.path, self.style, self.label_map,
                    self.data_mod, self.file_ext))

    @staticmethod
    def is_true_video(data_mod, file_ext):
        '''
        NOTE: currently only support RGB modality
        '''
        if (("RGB" == data_mod) and 
            (file_ext in __supported_video_files__[data_mod])):
            return(True)
        else:
            return(False)

    @staticmethod
    def strip_file_ext(path, ext):
        suffix = "." + ext
        if (__strict__):
            assert (suffix in path), "Incorrect path, should be *.<ext>"
            assert (suffix == path[-len(suffix):]), \
                "Incorrect path, file extension should be placed at the end"
        return(path[:-len(suffix)])

    @staticmethod
    def collect_samples(path, style, label_map, data_mod, file_ext):
        '''
        collect a list of samples = list of samples, while each sample
        = (video, relative path, class id, label)
        here, video may refers to a collection of images
        '''
        if ("UCF-101" == style):
            # get all labels
            labels = []
            for _label in os.listdir(path):
                if (os.path.isdir(os.path.join(path, _label))):
                    labels.append(_label)
            # TODO: check whether we need to sort it
            labels = sorted(labels)
            # get all videos' file paths and annotations
            # NOTE: each sample is a tuple = 
            # (vid_path, relative_addr, class_id, label)
            samples = []
            if (VideoCollector.is_true_video(data_mod, file_ext)):
                for _label in labels:
                    for _video in os.listdir(os.path.join(path, _label)):
                        _path = os.path.join(path, _label, _video)
                        _rpath = os.path.join(_label, _video)
                        _rpath = VideoCollector.strip_file_ext(_rpath, file_ext)
                        _sample = (_path, _rpath, label_map[_label], _label)
                        samples.append(_sample)
                # return results
                return(samples)
            else:
                for _label in labels:
                    for _video in os.listdir(os.path.join(path, _label)):
                        _path = os.path.join(path, _label, _video)
                        _rpath = os.path.join(_label, _video)
                        _sample = (_path, _rpath, label_map[_label], _label)
                        samples.append(_sample)
                # return results
                return(samples)                
        else:
            assert True, "Unsupported Dataset Struture Style"        

    def __get_samples__(self):
        return(self.samples)



weizmann_label_map = {
    'bend' : 1,
    'jack' : 2,
    'jump' : 3,
    'pjump': 4,
    'run'  : 5,
    'side' : 6,
    'skip' : 7,
    'walk' : 8,
    'wave1': 9,
    'wave2': 10
}





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


def _preprocess(task_dict):
    src_vid = task_dict['src_vid']
    tgt_vid = task_dict['tgt_vid']
    os.makedirs(tgt_vid)
    video.video2frames(src_vid, tgt_vid)

def preprocess(src_path, tgt_path, style, label_map):
    '''
    Single core pre-processing
    '''
    parser = VideoCollector(src_path, style, label_map, seek_file=True)
    samples = parser.__get_samples__()    
    for _sample in samples:
        src_vid = _sample[0]
        tgt_vid = os.path.join(tgt_path, _sample[1])
        _preprocess({'src_vid': src_vid, 'tgt_vid':tgt_vid})

def generate_preprocess_tasks(src_path, tgt_path, style, label_map):
    parser = VideoCollector(src_path, style, label_map, seek_file=True)
    samples = parser.__get_samples__()
    tasks = []  
    for _sample in samples:
        src_vid = _sample[0]
        tgt_vid = os.path.join(tgt_path, _sample[1])
        tasks.append({'src_vid': src_vid, 'tgt_vid':tgt_vid})
    return(tasks)

def task_executor(task_queue):
    while True:
        task = task_queue.get()
        if ('Done' == task):
            break
        _preprocess(task)

if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))

    raw_dataset = os.path.join(dir_path, 'Weizmann', 'Weizmann-raw')
    new_dataset = os.path.join(dir_path, 'Weizmann', 'Weizmann-img')
    # parser = VideoCollector(
    #     raw_dataset,
    #     'UCF-101', weizmann_label_map)
    # print(parser.__get_samples__())

    # preprocess(raw_dataset, new_dataset, 'UCF-101', weizmann_label_map)

    tasks = generate_preprocess_tasks(
        raw_dataset, new_dataset, 'UCF-101', weizmann_label_map)
    process_num = min(mp.cpu_count()+2, len(tasks)+1)

    task_queue = Queue()
    # Init process
    process_list = []
    for i in range(process_num):
        p = Process(target=task_executor, args=(task_queue,))
        p.start()
        process_list.append(p)

    for _task in tasks:
        task_queue.put(_task)
    for i in range(process_num):
        task_queue.put('Done')

    for p in process_list:
        p.join()
