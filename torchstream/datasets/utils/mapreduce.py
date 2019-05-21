"""
Providing simple helpers to manage batch processing tasks in a map-reduce way
with multiprocessing queue mechanism 
"""
import os
import copy
import time
import pickle
import multiprocessing as mp

from .__config__ import *
if __TQDM__:
    import tqdm


END_FLAG = "[DONE]"


class Worker(object):
    """
    """
    ## Constructor
    #  @param wid int: a worker id for this object
    #  @param mapper callable: function to be mapped to inputs
    #  It must map 1 source object to 1 target object
    #  @param reducer callable: function reducing the inputs
    #  It must works on a list which holds the results of
    #  mapper and returns 1 list (because a reducer might have reducing
    #  block size limit, and the reducer might only produce partial results
    #  which might be further reduced by the same reducer)
    #  @param retries int: retry number (if execution failed)
    def __init__(self, wid, mapper, reducer=None, **kwargs):
        """
        """
        self._wid = wid

        self._mapper = mapper
        self._reducer = reducer
        self._kwargs = copy.deepcopy(kwargs)
        self._results = []

        self._filename = "mapreduce.worker{}.{}.tmp.pkl".format(
            hash((self._wid, self._mapper, self._reducer)),
            int(time.time())
            )

    def __repr__(self, idents=0):
        header = idents * "\t"
        string = header + "Worker Object: ID-{}\n".format(self._wid)
        string += header + "[mapper] - {}\n".format(self._mapper.__name__)
        string += header + "[reducer] - {}\n".format(self._reducer.__name__)
        return string

    ## Calling
    #  
    #  @param task_queue: mp.Queue: input task queue
    def __call__(self, task_queue):
        while True:
            task = task_queue.get()
            if END_FLAG == task:
                break
            _result = self._mapper(**task, **(self._kwargs))
            self._results.append(_result)
            if self._reducer is not None:
                self._results = self._reducer(self._results)
        
        f = open(self._filename, "wb")
        pickle.dump(self._results, f)
        f.close()

    ## Reset
    #  
    #  Clean this worker's local results
    def reset(self):
        """
        Clean local results
        """
        self._results = []

    ## Get Results
    #  
    #  Get a copy of this worker's local results
    def get_results(self):
        """
        Get a copy of this worker's results
        """
        f = open(self._filename, "rb")
        _result = pickle.load(f)
        f.close()
        os.remove(self._filename)
        return(_result)


class Manager(object):
    """
    currently, we only support run a specified task once
    """
    def __init__(self, name, mapper, reducer=None, **kwargs):
        self.name = copy.deepcopy(name)
        self.mapper = mapper
        self.reducer = reducer
        self.kwargs = copy.deepcopy(kwargs)
        
        self.workers = []
        self.result = None

    def __repr__(self, idents=0):
        header = idents * "\t"
        string = header + "Manager Object: {}\n".format(self.name)
        string += header + "[mapper] - {}\n".format(self.mapper.__name__)
        string += header + "[reducer] - {}\n".format(self.reducer.__name__)
        string += header + "[workers] - {} homogeneous\n".\
            format(len(self.workers))
        return string

    def hire(self, worker_num):
        for _i in range(worker_num):  
            self.workers.append(Worker(
                wid=_i, mapper=self.mapper, reducer=self.reducer,
                **(self.kwargs)))

    ##  Launch a major task which contains many sub-tasks
    #   
    #   @param tasks iteratable: 
    def launch(self, tasks):
        """
        """
        num_workers = len(self.workers)
        assert num_workers > 0, "You cannot work with 0 workers!"

        num_tasks = len(tasks)
        assert num_tasks > 0, "You cannot launch null!"

        ## Limit Queue Size
        #  By limiting the size of task queue, we can make enquing & dequing
        #  nearly balanced. Thus, we can take the shown tqdm progress of 
        #  "putting tasks" as an accurate estimation of "processing tasks"
        task_queue = mp.Queue(2 * num_workers)


        ## init processes
        #  
        process_list = []
        for _worker in self.workers:
            p = mp.Process(target=_worker, args=(task_queue,))
            p.start()
            process_list.append(p)

        print("MANAGER : [{}] starting jobs".format(self.name))

        ## init tasks
        #  
        if __TQDM__:
            _tasks = tqdm.tqdm(tasks)
        else:
            _tasks = tasks
        for _task in _tasks:
            task_queue.put(_task)
        for i in range(num_workers):
            task_queue.put(END_FLAG)

        ## waiting for workers to join
        #  
        #  
        for p in process_list:
            p.join()


        ## aggregate results from workers
        #
        #
        print("MANAGER : [{}] aggregating".format(self.name))
        results = []
        if __TQDM__:
            _workers = tqdm.tqdm(self.workers)
        else:
            _workers = self.workers
        for _worker in _workers:
            _partial_results = _worker.get_results()
            results.extend(_partial_results)
            if self.reducer is not None:
                results = self.reducer(results)

        return results
