"""
Split filters
"""
__all__ = [
    "TrainsetFilter", "ValsetFilter", "TestsetFilter"
]

# for Weizmann, we use ALL SAMPLES for
# training, validation and testing

class TrainsetFilter(object):
    """
    Always True
    """
    def __init__(self):
        pass
    def __call__(self, sample):
        return True

class ValsetFilter(object):
    """
    Always True
    """
    def __init__(self):
        pass
    def __call__(self, sample):
        return True

class TestsetFilter(object):
    """
    Always True
    """
    def __init__(self):
        pass
    def __call__(self, sample):
        return True