class Meter(object):
    """Computes and stores the average, sum & current value
    """
    def __init__(self, name=""):
        self.name = name
        self.val = 0    # last value
        self.sum = 0    # sum
        self.count = 0  # weight count

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        """running avg
        """
        avg = self.sum / self.count
        return avg
    
    def __repr__(self):
        string = self.__class__.__name__ + " {}".format(self.name)
        string += "\n[val]:\t" + str(self.val)
        string += "\n[sum]:\t" + str(self.sum)
        string += "\n[count]:\t" + str(self.count)
        string += "\n[avg]:\t" + str(self.avg)
        return string
