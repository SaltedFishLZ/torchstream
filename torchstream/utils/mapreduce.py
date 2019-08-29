"""
Providing simple helpers to manage batch processing tasks in a map-reduce way
with multiprocessing queue mechanism
"""
__all__ = ["Worker", "Manager"]

import os
import copy
import time
import tqdm
import pickle
import logging
import multiprocessing as mp

from . import __config__

# NOTE: don't touch it now !!!
END_FLAG = "[DONE]"

# configuring logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(__config__.LOGGER_LEVEL)


class Worker(object):
    """
    Args:
        wid (int): worker id for this object
        mapper (callable): function to be mapped to inputs
            It must map 1 source object to 1 target object
        reducer (callable): function reducing the inputs
            It must works on a list which holds the results of
            mapper and returns 1 list (because a reducer might have reducing
            block size limit, and the reducer might produce partial results
            which might be further reduced by the same reducer)
        retries (int): retry number (if execution failed)
    """
    def __init__(self, wid, mapper, reducer=None):
        self._wid = wid
        self._mapper = mapper
        self._reducer = reducer

        self._results = []

        self._filename = "mapreduce.worker{}.{}.{}.tmp.pkl".format(
            self._wid,
            hash((self._wid, str(self._mapper), str(self._reducer))),
            int(time.time())
            )

    def __repr__(self, idents=0):
        header = idents * "\t"
        string = header + "Worker Object: ID-{}\n".format(self._wid)
        string += header + "[mapper] - {}\n".format(self._mapper.__name__)
        string += header + "[reducer] - {}\n".format(self._reducer.__name__)
        return string

    def __call__(self, task_queue):
        """
        Args:
            task_queue (mp.Queue): input task queue
        """
        while True:
            task = task_queue.get()
            if END_FLAG == task:
                break
            _result = self._mapper(**task)
            self._results.append(_result)
            if self._reducer is not None:
                self._results = self._reducer(self._results)
        # currently, use pickle to dump temporary results
        with open(self._filename, "wb") as f:
            pickle.dump(self._results, f)

    def reset(self):
        """Clean this worker's local results
        """
        self._results = []

    def get_results(self):
        """Get a copy of this worker's local results
        """
        if not os.path.exists(self._filename):
            if __config__.STRICT:
                raise ValueError("local result missing!")
            else:
                return None
        with open(self._filename, "rb") as f:
            _result = pickle.load(f)
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
            self.workers.append(Worker(wid=_i,
                                       mapper=self.mapper,
                                       reducer=self.reducer))

    def launch(self, tasks, progress=False):
        """
        Args:
            tasks (iteratable): a major task which contains many sub-tasks
            progress (bool): using TQDM to show progress
        """
        num_workers = len(self.workers)
        assert num_workers > 0, "You cannot work with 0 workers!"

        num_tasks = len(tasks)
        assert num_tasks > 0, "You cannot launch empty task!"

        # Limit Queue Size
        # By limiting the size of task queue, we can make enquing & dequing
        # nearly balanced. Thus, we can take the shown tqdm progress of
        # "putting tasks" as an accurate estimation of "processing tasks"
        task_queue = mp.Queue(2 * num_workers)

        # init processes
        process_list = []
        for _worker in self.workers:
            p = mp.Process(target=_worker, args=(task_queue,))
            p.start()
            process_list.append(p)

        # start jobs
        logger.info("MANAGER: [{}] starting jobs".format(self.name))

        # init tasks
        if progress:
            _tasks = tqdm.tqdm(tasks)
        else:
            _tasks = tasks
        for _task in _tasks:
            task_queue.put(_task)
        for i in range(num_workers):
            task_queue.put(END_FLAG)

        # waiting for workers to join
        info_str = "MANAGER: [{}] waiting for late workers".format(self.name)
        logger.info(info_str)

        for p in process_list:
            p.join()

        # aggregate results from workers
        info_str = "MANAGER: [{}] aggregating".format(self.name)
        logger.info(info_str)

        results = []
        if progress:
            _workers = tqdm.tqdm(self.workers)
        else:
            _workers = self.workers
        for _worker in _workers:
            _partial_results = _worker.get_results()
            results.extend(_partial_results)
            if self.reducer is not None:
                results = self.reducer(results)

        return results
