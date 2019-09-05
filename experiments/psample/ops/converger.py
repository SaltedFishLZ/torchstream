import importlib

import torch

class LinearLossConverger(torch.nn.Module):
    """
    Args:
        loss (str): critetrion name,
        currently, 1 loss for all branches.
    """
    def __init__(self, loss_name=None, loss_params={}):
        super(LinearLossConverger, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        if loss_name is not None:
            loss_class = getattr(torch.nn, loss_name)
            self.loss = loss_class(**loss_params)

    def forward(self, predicts, target):
        total_loss = 0
        for pred in predicts:
            total_loss += self.loss(pred, target)
        return total_loss

if __name__ == "__main__":
    CLS_NUM = 127
    N = 10

    probs = torch.randn(N, CLS_NUM)
    target = torch.max(probs, dim=1)[1]

    predicts = []
    for i in range(4):
        predicts.append(
            probs.clone().requires_grad_(True)
            # torch.randn(N, CLS_NUM, requires_grad=True)
            )


    critetrion = LinearLossConverger()
    loss = critetrion(predicts, target)
    print(loss)
    loss.backward()
