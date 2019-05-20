
# for this dataset, we use ALL SAMPLES for 
# training, validation and testing

class TrainsetFilter(object):
    def __init__(self, split=None):
        pass
    def __call__(self, sample):
        return(True)

class ValsetFilter(object):
    def __init__(self, split=None):
        pass
    def __call__(self, sample):
        return(True)

class TestsetFilter(object):
    def __init__(self, split=None):
        pass
    def __call__(self, sample):
        return(True)