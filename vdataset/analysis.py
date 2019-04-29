import os
import sys
import copy
import time
import pickle
import logging
import importlib
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from .__init__ import __verbose__, __vverbose__, __test__, __strict__
from .dataset import VideoDataset

LOG_INTERVAL = 10

def varray_sum_raw(varray):
    # get frame shape
    (_t, _h, _w) = varray.shape[0:3]
    nums = _t * _h * _w

    # Numpy sum over multiple axes
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    sums = varray.sum(axis=(0,1,2))

    return(sums, nums)

def varray_sum_rsq(varray, means):
    (_t, _h, _w) = varray.shape[0:3]
    nums = _t * _h * _w

    residuals = varray - np.tile(means, (_t, _h, _w, 1))
    rsquares = np.square(residuals)
    # Numpy sum over multiple axes
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    sums = rsquares.sum(axis=(0,1,2))

    return(sums, nums)


# ---------------------------------------------------------------- #
#               Collect Samples' Shape Information                 #
# ---------------------------------------------------------------- #
                                                                    
# worker function
def get_shape(worker_id, vid_dataset, task_queue, result_queue):
    '''
    - worker_id : unique worker indentifier for each worker function
    - vid_dataset : a VideoDataset object to be analyzed
    - task_queue : mp.Queue object for input tasks (dataset sample id)
    - result_queue : mp.Queue object for results (dumped pickle file name)
    '''
    worker_id_str = "{0:05d}".format(worker_id)
    # init local results
    process_shapes = []
    # local loop
    while True:
        task = task_queue.get()
        if ("DONE" == task):
            # tasks all finished
            if (__verbose__):
                info_str = "WORKER-{} : [get_shape] done".\
                    format(worker_id_str)
                logging.info(info_str)
                if (__vverbose__):
                    print(info_str)
            # break the working loop
            break
        else:
            # main job
            varray, cid = vid_dataset.__getitem__(task)
            process_shapes.append(varray.shape)
            # output status
            if (task % LOG_INTERVAL == 0):
                info_str = "WORKER-{} : [get_shape] progress [{}/{}]".\
                    format(worker_id_str, task, vid_dataset.__len__())
                print(info_str)
    # local tasks done
    print("DEBUG: task queue size {}".format(task_queue.qsize()))
    # in Python 3.x, use "int" instead of "long"
    ticks = int(time.time() * 100000)
    print("DEBUG: ticks {}".format(ticks))
    pkl_name = "process_{}_{}.shapes.tmp.pkl".format(worker_id_str, ticks)
    f = open(pkl_name, "wb")
    pickle.dump(process_shapes, f)
    f.close()
    # return the pickle file name
    result_queue.put(pkl_name)

# multi-process wrapper
def get_shapes(vid_dataset, num_proc):
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    # init process
    process_list = []
    for _i in range(num_proc):
        p = mp.Process(target=get_shape, \
            args=(int(_i), vid_dataset, task_queue, result_queue))
        p.start()
        process_list.append(p)
    # init tasks
    print("MANAGER : [get_shapes] start")
    tasks = list(range(vid_dataset.__len__()))
    for _task in tasks:
        task_queue.put(_task)
    for i in range(num_proc):
        task_queue.put("DONE")
    # waiting for join
    for p in process_list:
        p.join()
    # dump reulsts from result queue
    info_str = "MANAGER : [get_shapes]: dumping results"
    print(info_str)
    shapes = []
    result_queue.put("END")
    
    while True:
        ret = result_queue.get()
        if (ret == "END"):
            break
        f = open(ret, "rb")
        partial_shapes = pickle.load(f)
        shapes.extend(partial_shapes)
        f.close()
        os.remove(ret)
        
    # dump result to file & remove temporaray files
    pkl_name = "{}.shapes.pkl".format(vid_dataset.dataset)
    info_str = "MANAGER : [get_shapes]: pickle dumping to \"{}\""\
        .format(pkl_name)
    print(info_str)
    f = open(pkl_name, "wb")
    pickle.dump(shapes, f)
    f.close()
    # return results
    print("MANAGER : [get_shapes]: total {} shapes".format(len(shapes)))
    return(shapes)



# ---------------------------------------------------------------- #
#                Collect Samples' Pixel Mean Value                 #
# ---------------------------------------------------------------- #
                                                                    
# worker function
def get_sum_raw(vid_dataset, task_queue, result_queue):
    
    # init local results
    process_sums = np.array([0.0, 0.0, 0.0])
    process_nums = 0.0    

    # local loop
    while True:
        task = task_queue.get()

        if ("DONE" == task):
            
            # tasks all finished
            if (__verbose__):
                info_str = "[get_shape] {} done".format(mp.current_process())
                logging.info(info_str)
                if (__vverbose__):
                    print(info_str)
            break
        
        else:
            # main job            
            varray, cid = vid_dataset.__getitem__(task)
            sums, nums = varray_sum_raw(varray)

            process_sums += sums
            process_nums += nums

            # logging progress
            if (task % LOG_INTERVAL == 0):
                print("[get_sum_raw] progress [{}/{}]".\
                    format(task, vid_dataset.__len__()))    

    # local tasks done
    print("DEBUG: task queue size {}".format(task_queue.qsize()))
    result_queue.put((sums, nums))

# multi-process wrapper
def get_means(vid_dataset, num_proc):
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # init process
    process_list = []
    for _i in range(num_proc):
        p = mp.Process(target=get_sum_raw, \
            args=(vid_dataset, task_queue, result_queue))
        p.start()
        process_list.append(p)
    
    # init tasks
    print("[get_means] start")
    tasks = list(range(vid_dataset.__len__()))
    for _task in tasks:
        task_queue.put(_task)
    for i in range(num_proc):
        task_queue.put("DONE")
    
    # waiting for join
    for p in process_list:
        p.join()
    
    # aggregate
    print("[get_means] aggregating")
    sums = np.array([0.0, 0.0, 0.0])
    nums = 0.0
    result_queue.put("END")
    while True:
        result = result_queue.get()
        if ("END" == result):
            break
        sums += result[0]
        nums += result[1]
    means = sums / nums
    # dump result to file
    print("[get_means] dumping results")    
    f = open("{}.means.pkl".format(vid_dataset.dataset), "wb")
    pickle.dump(means, f)
    f.close()
    print(means)
    # return results
    return(means)




# ---------------------------------------------------------------- #
#              Collect Samples' Pixel Variance Value               #
# ---------------------------------------------------------------- #
                                                                    
def get_sum_rsq(vid_dataset, means, task_queue, result_queue):
    while True:
        task = task_queue.get()
        if ("DONE" == task):
            break
        if (task % LOG_INTERVAL == 0):
            print("[get_sum_rsq] progress [{}/{}]".\
                format(task, vid_dataset.__len__()))
        varray, cid = vid_dataset.__getitem__(task)
        sums, nums = varray_sum_rsq(varray, means)
        result_queue.put((sums, nums))

def get_vars(vid_dataset, means, num_proc):
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    # init process
    process_list = []
    for _i in range(num_proc):
        p = mp.Process(target=get_sum_rsq, \
            args=(vid_dataset, means, task_queue, result_queue,))
        p.start()
        process_list.append(p)
    # init tasks
    print("[get_vars] start")
    tasks = list(range(vid_dataset.__len__()))
    for _task in tasks:
        task_queue.put(_task)
    for i in range(num_proc):
        task_queue.put("DONE")
    # waiting for join
    for p in process_list:
        p.join()
    # aggregate
    print("[get_vars] aggregating")
    sums = np.array([0.0, 0.0, 0.0])
    nums = 0.0
    for i in range(2 * num_proc):
        result_queue.put("END")
    while True:
        result = result_queue.get()
        if ("END" == result):
            break
        sums += result[0]
        nums += result[1]
    vars = sums / nums
    # dump result to file
    print("[get_vars] dumping results")
    f = open("{}.vars.pkl".format(vid_dataset.dataset), "wb")
    pickle.dump(vars, f)
    f.close()
    # return results
    print(vars)
    # return results
    return(vars)

def test_functions():
    from .video import video2ndarray
    dir_path = os.path.dirname(os.path.realpath(__file__))
    vid_path = os.path.join(dir_path, "test.avi")
    # read video to varray
    varray = video2ndarray(vid_path,
            color_in="BGR",
            color_out="RGB")
    print(varray.shape)
    
    sums, nums = varray_sum_raw(varray)
    print(sums / nums)





if __name__ == "__main__":
    # test_functions()

    # DATASET = "Weizmann"
    for DATASET in ["Weizmann", "HMDB51"]:
        print("")
        print("################################")
        print("Analyzing {} ...".format(DATASET))
        print("################################")
        print("")
        dset = importlib.import_module(
            "vdataset.{}".format(DATASET))
        allset = VideoDataset(
                dset.prc_data_path, DATASET, split="1")
        shapes = get_shapes(allset, 32)
        # means = get_means(allset, 32)
        # vars = get_vars(allset, means, 32)

    # f = open("{}.shapes.pkl".format(DATASET), "rb")
    # shapes = pickle.load(f)
    # f.close()
    # (lengths, heights, widths, channels) = zip(*shapes)
    # # print(lengths)
    # plt.hist(lengths, density=True, bins=10)
    # plt.show()
