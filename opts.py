import argparse

parser = argparse.ArgumentParser(description="PyTorch Video Recognition Template")

# configuration file
parser.add_argument("config", type=str,
                    help="path to configuration file")
parser.add_argument('--gpus', nargs='+', type=int, default=None)

# Training Configs (Will override the JSON configuration if there is a conflict)
parser.add_argument("-b", "--batch-size", default=64, type=int, metavar='N',
                    help="mini-batch size")
parser.add_argument("--epochs", default=20, type=int, metavar='N',
                    help="number of total epochs to run")
parser.add_argument("--lr", "--learning-rate", default=0.001, type=float,
                    metavar="LR", help="initial learning rate")
