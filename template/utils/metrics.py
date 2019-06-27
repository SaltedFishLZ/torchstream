import torch
import sklearn
import numpy as np
import matplotlib.pyplot as plt

def output2pred(output, maxk):
    """
    Args:
        maxk (int): maximum of topk
        output (Tensor): [N][K], network classfication probability
        return: (LongTensor): [maxk][N], prediction tensor
    """
    # pred [N][MaxK] -> [MaxK][N]
    _, pred = output.data.topk(maxk, dim=1)
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
    # target [N] -> [1][N] -> [MaxK][N]
    corrects = pred.eq(target.unsqueeze(dim=0).expand_as(pred))
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

        pred = output2pred(output.detach(), self.maxk)
        corrects = classify_corrects(pred, target)

        res = {}
        for k in self.topk:
            acc_topk = float(corrects[k-1].float().sum().item()) / N * 100.0
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

        pred = output2pred(output.detach(), self.maxk)
        corrects = classify_corrects(pred, target)

        if self.corrects is not None:
            corrects |= self.corrects

        self.corrects = corrects

        res = {}
        for k in self.topk:
            acc_topk = float(corrects[k-1].float().sum().item()) / N * 100.0
            res[k] = acc_topk
        return res

def confusion_matrix(predicts, targets, normalize=False):
    """
    Args:
        predicts: prediction
        targets: ground truth
        normalized each entry to [0, 1]
    """
    cm = sklearn.metrics.confusion_matrix(targets, predicts)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm

def plot_confusion_matrix(cm, labels=None, inverse_label=False,
                          title="Confusion Matrix",
                          cmap=plt.cm.Blues):
    """
    """
    classes = list(range(cm.shape[0]))
    if labels is not None:
        if inverse_label:
            d = {v: k for k, v in labels.items()}
            labels = d
        classes = [labels[i] for i in classes]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if cm.dtype == np.float:
        fmt = ".2f"
    else:
        fmt = "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()