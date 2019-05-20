"""
"""
import os
import sys
import copy
import logging
import operator
import importlib

from vdataset import constant
from vdataset import metasets
from vdataset import metadata
from vdataset import preprocess
from vdataset import mapreduce

def main(name):
    mset = importlib.import_module("vdataset.metasets.{}".format(name))
    collector = metadata.Collector(mset.RAW_DATA_PATH, mset,
                                   ext="webm")
    src_sample_set = collector.collect_samples()
    dst_sample_set = src_sample_set.root_migrated(mset.PRC_DATA_PATH)
    
    src_sample_list = src_sample_set.get_samples()
    src_sample_list.sort()

    dst_sample_list = dst_sample_set.get_samples()
    dst_sample_list.sort()


    print("Santity Check")
    _pairs = list(zip(src_sample_list, dst_sample_list))
    for _i in _pairs:
        _i[1].to_video(ext="avi")
        if _i[0].name != _i[1].name:
            print("{} - {}".format(_i[0].name, _i[1].name))
            exit(1)

    print("Main Jobs")
    tasks = []
    for _i in _pairs:
        tasks.append({"src_sample" : _i[0], "dst_sample" : _i[1]})

    manager = mapreduce.Manager(name="slicing-{}".format(name),
                                mapper=preprocess.vid2vid,
                                retries=10
                                )
    manager.hire(worker_num=10)
    result = manager.launch(tasks=tasks)
    
    for _i, _status in enumerate(result):
        if not _status:
            print("Task-[{}] Failed".format(_i))
            print(tasks[_i])

if __name__ == "__main__":
    main("sth_sth_v2")
