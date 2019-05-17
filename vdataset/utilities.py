# -*- coding: utf-8 -*-
import time

class Timer(object):
    '''
    A simple Timer which can start, pause, reset and
    get total time.
    NOTE : Unit is second(s)
    - acc_time : accumulated time (s)
    - counting : timer state
    - tick : physical time of the latest starting point
    '''
    def __init__(self):
        self.acc_time = 0
        self.counting = False
        self.tick = 0

    def start(self):
        if (False == self.counting):
            self.counting = True
            self.tick = time.time()
        else:
            pass

    def pause(self):
        if (True == self.counting):
            self.counting = False
            duration = time.time() - self.tick
            self.acc_time += duration
        else:
            pass
    
    def reset(self):
        self.acc_time = 0
        self.counting = False
    
    def report(self):
        return(self.acc_time)


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

