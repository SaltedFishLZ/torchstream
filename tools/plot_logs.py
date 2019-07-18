import re
import sys

import numpy as np
import matplotlib.pyplot as plt

def parse_loss(fpath, epoch_num=None):
    losses = []
    loss_regex = r"Loss:\s+\d+\.\d+\s*\(\s*\d+\.\d+\)"
    if epoch_num is None:
        epoch_count = 0

    iter_count = 0
    with open(fpath, "r") as fin:
        for line in fin:
            if "Validation" in line:
                if epoch_num is None:
                    epoch_count += 1
                continue
            loss_str = re.findall(loss_regex, line)
            if len(loss_str) == 1:
                loss_str = loss_str[0]
                running_loss = re.findall(r"\d+\.\d+", loss_str)[1]
                running_loss = float(running_loss)
                losses.append(running_loss)
                iter_count += 1
    losses = np.array(losses)

    if epoch_num is not None:
        epoch_count = epoch_num
    epoches = np.linspace(0, epoch_count - 1, num=iter_count)

    return epoches, losses


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        log_file = sys.argv[1]
        epoch_num = None
        if len(sys.argv) == 3:
            epoch_num = int(sys.argv[2])
        epochs, losses = parse_loss(log_file, epoch_num)
        print(losses)
        print(len(losses))
        plt.plot(epochs, losses, "-b", label="Loss")
        plt.xlabel(r"# Epochs")
        plt.ylabel(r"Loss")
        plt.show()
    else:
        print("cmd <log file>")
