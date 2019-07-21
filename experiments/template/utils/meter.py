class Meter(object):
    """Computes and stores the average, sum & current value
    """
    def __init__(self, name=""):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0    

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __repr__(self):
        string = self.__class__.__name__ + " {}".format(self.name)
        string += "\n[val]:\t" + str(self.val)
        string += "\n[sum]:\t" + str(self.sum)
        string += "\n[count]:\t" + str(self.count)
        string += "\n[avg]:\t" + str(self.avg)
        return string
