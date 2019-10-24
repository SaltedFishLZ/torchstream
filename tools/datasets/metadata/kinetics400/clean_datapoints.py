import pickle
import argparse

from torchstream.io.datapoint import DataPoint
import torchstream.io.backends.opencv as backend

parser = argparse.ArgumentParser()
parser.add_argument("src", type=str, help="path to source datapoint")
parser.add_argument("dst", type=str, help="path to destination datapoint")
parser.add_argument("--root", type=str, default=None,
                    help="path to dataset root")


def remove_empty_videos(datapoints, root=None):
    cleanpoints = []
    for datapoint in datapoints:
        assert isinstance(datapoint, DataPoint)
        if root is not None:
            datapoint.root = root
            datapoint._path = datapoint.path

        if datapoint.seq:
            if len(datapoint.framepaths) > 31:
                cleanpoints.append(datapoint)
            else:
                print(datapoint)
        else:
            vpath = datapoint.path
            varray = backend.video2ndarray(vpath)
            t, h, w, c = varray.shape
            if t > 31:
                cleanpoints.append(datapoint)
            else:
                print(datapoint)
    return cleanpoints


if __name__ == "__main__":
    args = parser.parse_args()

    datapoints = []
    with open(args.src, "rb") as f:
        datapoints = pickle.load(f)

    datapoints = remove_empty_videos(datapoints, args.root)

    with open(args.dst, "wb") as f:
        pickle.dump(datapoints, f)
