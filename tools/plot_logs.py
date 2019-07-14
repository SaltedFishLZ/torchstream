import re
import sys

def parse_loss(fpath):
    losses = []
    loss_regex = r"Loss:\s+\d+\.\d+\s*\(\s*\d+\.\d+\)"
    with open(fpath, "r") as fin:
        for line in fin:
            loss_str = re.findall(loss_regex)
            if len(loss_str) == 1:
                running_loss = re.findall(r"\d+\.\d+", loss_str)[1]
                running_loss = float(running_loss)
                print(running_loss)
                losses.append(running_loss)
    print(len(running_loss))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for log_file in sys.argv[1: -1]:
            parse_loss(log_file)
    else:
        print("cmd <log 0> <log 1> ...")