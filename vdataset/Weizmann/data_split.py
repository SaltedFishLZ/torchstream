
# for this dataset, we use ALL SAMPLES for 
# training, validation and testing

class for_train(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        return(True)

class for_val(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        return(True)

class for_test(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        return(True)