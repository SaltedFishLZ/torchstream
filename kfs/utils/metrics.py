import torch

def output2pred(output, maxk):
    """
    Args:
        maxk (int): maximum of topk
        output (Tensor): [N][K], network classfication probability
        return: (LongTensor): [maxk][N], prediction tensor
    """
    # pred [N][MaxK] -> [MaxK][N]
    _, pred = output.topk(maxk, dim=1)
    ## DEBUG
    # print("prediction", pred)
    pred = pred.permute(1, 0)
    return pred


def classify_corrects(pred, target):
    """
    Args:
        pred (LongTensor): [K][N] predicted class
        target (LongTensor): [N] target class
        return (ByteTensor): [K][N] bit mask, indicates topk hits or not
    """
    maxk = pred.size(0)
    # target [N] -> [1][N]
    corrects = pred.eq(target.view(1, -1).expand_as(pred))
    for k in range(1, maxk):
        corrects[k] = corrects[k - 1] | corrects[k]
    return corrects



class ClassifyAccuracy(object):
    """
    Args:
        topk (tuple): Ks
    """
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.maxk = max(topk)

    def __call__(self, output, target):
        """
        Args:
            output (Tensor): [N][M] Sample-N, Class-M
            target (LongTensor): [N] Sample-N
        """
        N, M = output.size()
        assert (N,) == tuple(target.size()), ValueError("Incorrect target shape")

        pred = output2pred(output, self.maxk)
        corrects = classify_corrects(pred, target)

        res = {}
        for k in self.topk:
            acc_topk = float(corrects[k-1].sum()) / N * 100.0
            res[k] = acc_topk
        return res



class MultiChanceClassifyAccuracy(object):
    """
    Args:
        topk (tuple): Ks
    """
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.maxk = max(topk)
        self.chances = 0
        self.corrects = None

    def reset(self):
        """Reset history
        """
        self.chances = 0
        self.corrects = None

    def __call__(self, output, target):
        """
        Args:
            output (Tensor): a tensor
            target
        """
        N, M = output.size()
        assert (N,) == tuple(target.size()), ValueError("Incorrect target shape")

        pred = output2pred(output, self.maxk)
        corrects = classify_corrects(pred, target)

        if self.corrects is not None:
            corrects |= self.corrects

        self.corrects = corrects

        res = {}
        for k in self.topk:
            acc_topk = float(corrects[k-1].sum()) / N * 100.0
            res[k] = acc_topk
        return res
