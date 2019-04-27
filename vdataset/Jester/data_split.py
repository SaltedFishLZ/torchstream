import os
import pandas

from .common import trainset_df, valset_df, testset_df


class TrainsetFilter(object):
    def __init__(self):
        self.trainset = set()
        for idx, row in trainset_df.iterrows():
            video = row["video"]
            self.trainset.add(video)
    def __call__(self, sample):
        return(sample.file in self.trainset)

class ValsetFilter(object):
    def __init__(self):
        self.valset = set()
        for idx, row in valset_df.iterrows():
            video = row["video"]
            self.valset.add(video)
    def __call__(self, sample):
        return(sample.file in self.valset)

class TestsetFilter(object):
    def __init__(self):
        self.testset = set()
        for idx, row in testset_df.iterrows():
            video = row["video"]
            self.testset.add(video)
    def __call__(self, sample):
        return(sample.file in self.testset)