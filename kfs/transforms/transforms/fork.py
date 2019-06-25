import sys
import collections

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class Fork(object):
    """
    Args:
        transforms: an iterable, containing transform for each
            branch in the input data iterable.

        return: a tuple
    """
    def __init__(self, transforms):
        assert isinstance(transforms, Sequence), TypeError
        self.transforms = transforms

    def __repr__(self):
        string = self.__class__.__name__
        string += "\ntransforms in each branch:\n"
        for t in self.transforms:
            string += "\t-{}\n".format(t)
        return string

    def __call__(self, inputs):
        """
        """
        # # debug
        # print("DEBUG: fork used")
        assert isinstance(inputs, Sequence), TypeError
        assert len(inputs) == len(self.transforms), ValueError("Size mismatching")
        outputs = []
        N = len(self.transforms)
        for i in range(N):
            outputs.append(self.transforms[i](inputs[i]))
        return tuple(outputs)


class Join(object):
    """
    """

    def __init__(self, reducer):
        self.reducer = reducer
        raise NotImplementedError
    
    def __call__(self, inputs):
        """
        """
        return self.reducer(inputs)
