__test__    =   True
__strict__  =   True
__verbose__ =   True
__vverbose__=   True

import os
import sys
import copy
import logging
import multiprocessing as mp

from vdataset import video
from vdataset.dataset import VideoCollector
from vdataset.label_map import *

def preprocess(task_dict):
    src_vid = task_dict['src_vid']
    tgt_vid = task_dict['tgt_vid']
    if (not os.path.exists(tgt_vid)):
        try:
            os.makedirs(tgt_vid)
        except FileExistsError:
            pass
    ret = video.video2frames(src_vid, tgt_vid)
    return(ret)

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
        if (None == task):
            break
        ret = preprocess(task)
        # retry it
        cnt = 10
        while ((not ret) and (cnt > 0)):
            info_str = "Retry task {}".format(task)
            logging.info(info_str)
            if (__vverbose__):
                print(info_str)
            ret = preprocess(task)
            cnt -= 1
        if (not ret):
            print("Task {} failed after 10 trails!!!".format(task))




if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))

    raw_dataset = os.path.join("/home/zheng/Datasets", 'UCF101', 'UCF101-raw')
    new_dataset = os.path.join("/home/zheng/Datasets", 'UCF101', 'UCF101-img')

    tasks = generate_preprocess_tasks(
        raw_dataset, new_dataset, 'UCF101', label_maps['UCF101'])
    process_num = min(mp.cpu_count()*6, len(tasks)+1)

    task_queue = mp.Queue()
    # Init process
    process_list = []
    for _i in range(process_num):
        p = mp.Process(target=task_executor, args=(task_queue,))
        p.start()
        process_list.append(p)

    for _task in tasks:
        task_queue.put(_task)
    for i in range(process_num):
        task_queue.put(None)

    for p in process_list:
        p.join()
