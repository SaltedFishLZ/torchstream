import numpy as np
import torch.utils.data as data


class FakeData(data.Dataset):
    """
    Args:
        shape (tuple): (T, H, W, C)
        class_num (int >1): number of classes
        samples_perclass: samples for each class
    """
    def __init__(self, shape, class_num=10, samples_perclass=10,
                 transform=None, target_transform=None
                 ):

        self.shape = shape
        self.class_num = samples_perclass
        self.samples_perclass = samples_perclass
        self.transform = transform
        self.target_transform = target_transform

        self.lbls = list(range(class_num))
        self.samples = []
        for i in range(class_num):
            self.samples.extend(samples_perclass * [i])

    def __len__(self):
        return self.class_num * self.samples_perclass

    def __getitem__(self, idx):
        """
        """
        _cid = self.samples[idx]
        _blob = np.full(self.shape, _cid)

        if self.transform is not None:
            _blob = self.transform(_blob)
        if self.target_transform is not None:
            _cid = self.target_transform(_cid)
        return (_blob, _cid)
