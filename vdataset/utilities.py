# -*- coding: utf-8 -*-
import os
import time
import platform

class Timer(object):
    '''
    A simple Timer which can start, pause, reset and
    get total time.
    NOTE : Unit is second(s)
    - acc_time : accumulated time (s)
    - counting : timer state
    - tick : physical time of the latest starting point
    '''
    ## Documentation for a method.
    #  @param self The object pointer.    
    def __init__(self):
        self.acc_time = 0
        self.counting = False
        self.tick = 0

    ## Documentation for a method.
    #  @param self The object pointer.
    def start(self):
        if (False == self.counting):
            self.counting = True
            self.tick = time.time()
        else:
            pass

    ## Documentation for a method.
    #  @param self The object pointer.
    def pause(self):
        if (True == self.counting):
            self.counting = False
            duration = time.time() - self.tick
            self.acc_time += duration
        else:
            pass

    ## Documentation for a method.
    #  @param self The object pointer.
    def reset(self):
        self.acc_time = 0
        self.counting = False

    ## Documentation for a method.
    #  @param self The object pointer.
    def report(self):
        return(self.acc_time)



def strip_extension(file_name):
    return(os.path.splitext(file_name)[0])

def strip_path(file_path):
    return(os.path.basename(file_path))

def modification_date(path_to_file):
    return(os.path.getmtime(path_to_file))

def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    stat = os.stat(path_to_file)
    try:
        return stat.st_birthtime
    except AttributeError:
        # We're probably on Linux. No easy way to get creation dates here,
        # so we'll settle for when its content was last modified.
        return stat.st_mtime



if __name__ == "__main__":
    timer = Timer()
    timer.start()
    time.sleep(1)
    timer.pause()
    print(timer.report())
    timer.start()
    time.sleep(2)
    print(timer.report())
    timer.pause()
    print(timer.report())

