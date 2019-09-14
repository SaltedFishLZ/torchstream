"""
Author: Bernie Wang
"""
import os
import pickle
import csv

import torchstream.io.backends.opencv as backend
from torchstream.io.datapoint import DataPoint, UNKNOWN_LABEL
from torchstream.utils.mapreduce import Manager


KINETICS_METADATA_DIR = "/dnn/data/ActivityNet/Crawler/Kinetics/data/"
KINETICS_TRAIN_CSV = os.path.join(KINETICS_METADATA_DIR,
                                  "kinetics-400_train.csv")
KINETICS_TEST_CSV = os.path.join(KINETICS_METADATA_DIR,
                                 "kinetics-400_test.csv")
KINETICS_VAL_CSV = os.path.join(KINETICS_METADATA_DIR,
                                "kinetics-400_val.csv")

DOWNLOAD_SERVER_PREFIX = os.path.expanduser(
    "~/video-acc/download/torchstream/datasets/kinetics400/"
    )
KINETICS_LABEL_FILE = os.path.join(DOWNLOAD_SERVER_PREFIX,
                                   "kinetics400_labels.txt")

KINETICS_DATA_DIR = "/dnn/data/Kinetics/Kinetics-400-mp4"
KINETICS_TRAIN_DATA_DIR = os.path.join(KINETICS_DATA_DIR, "train")
KINETICS_TEST_DATA_DIR = os.path.join(KINETICS_DATA_DIR, "test")
KINETICS_VAL_DATA_DIR = os.path.join(KINETICS_DATA_DIR, "val")

KINETICS_PICKLE_DIR = os.path.expanduser(
    "~/video-acc/download/torchstream/datasets/kinetics400"
    )
PICKLE_TRAIN_FILE = os.path.join(KINETICS_PICKLE_DIR, "kinetics400_mp4_training.pkl")
PICKLE_TEST_FILE = os.path.join(KINETICS_PICKLE_DIR, "kinetics400_mp4_testing.pkl")
PICKLE_VAL_FILE = os.path.join(KINETICS_PICKLE_DIR, "kinetics400_mp4_validation.pkl")

# TRAIN_CORRUPT_INDICES = [1139, 6112, 9486, 18162, 23131, 27903, 35247, 39514, 49851, 60177, 61523, 74810, 86195, 105111, 109340, 117082, 117257, 127920, 133607, 134597, 134660, 136602, 146561, 147639, 154674, 157595, 183509, 184997, 191015, 197140, 197402, 217717, 219969]
# TEST_CORRUPT_INDICES = [14, 15, 22, 23, 41, 43, 57, 62, 65, 66, 68, 73, 79, 3604, 5404, 12957, 15328, 18970, 29697, 36448, 37647, 57586, 58168, 58397, 60725, 76615]
# VAL_CORRUPT_INDICES = [12809, 14334]

vid2filename = {}


def write_classes():
    if not os.path.exists(KINETICS_LABEL_FILE):
        classes = []
        for label_class in os.listdir(KINETICS_TRAIN_DATA_DIR):
            classes.append(label_class)

        classes.sort()

        with open(KINETICS_LABEL_FILE, "w") as f:
            for i, cls in enumerate(classes):
                f.write("{} {}\n".format(i+1, cls))


def init_cache(data_dir):
    for label_class in os.listdir(data_dir):
        for i, filename in enumerate(os.listdir(os.path.join(data_dir, label_class))):
            vid2filename[filename[:-18]] = filename.split('.')[0]


def parse_kinetics_csv(filename, corrupt_files=[]):
    labels = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # if i not in corrupt_files:
            vid = row["youtube_id"]
            label = row["label"] if "label" in row.keys() else UNKNOWN_LABEL
            labels.append((vid, label))
    return labels


def populate_datapoints(label_file, split="train", corrupt_files=[]):
    print("populating datapoints:", label_file)
    labels = parse_kinetics_csv(label_file, corrupt_files=corrupt_files)

    datapoints = []
    for vid, label in labels:
        if vid in vid2filename:
            if split == "train":
                rel_path = os.path.join("train", label)
            elif split == "val":
                rel_path = os.path.join("val", label)
            elif split == "test":
                rel_path = os.path.join("test", "test")
            else:
                print("split not valid")
                return

            datapoints.append(DataPoint(KINETICS_DATA_DIR,
                                        rel_path,
                                        vid2filename[vid],
                                        ext="mp4",
                                        label=label))

    return datapoints


def clean_datapoints(datapoints):
    """remove corrupt datapoints
    """
    def is_corrput(dp):
        assert isinstance(dp, DataPoint), TypeError
        if backend.video2ndarray(dp.path) is None:
            return [dp, ]
        else:
            return []

    def aggregate_list(list_of_list):
        assert isinstance(list_of_list, list), TypeError
        ret = []
        for l in list_of_list:
            ret += l
        return ret

    manager = Manager(name="clean Kinetics400 datapoints",
                      mapper=is_corrput,
                      reducer=aggregate_list,
                      retries=10)
    manager.hire(worker_num=64)

    tasks = []
    for dp in datapoints:
        tasks.append({"dp": dp})
    corruptpoints = manager.launch(tasks=tasks, progress=True)

    print(corrputpoints)
    corrputnames = [dp.name for dp in corruptpoints]

    cleanpoints = []
    for idx, dp in enumerate(datapoints):
        if dp.name in corruputnames:
            print("Corrupt ID", idx)
            print(dp)
        else:
            cleanpoints.append(dp)

    return cleanpoints


if __name__ == "__main__":
    write_classes()
    init_cache(KINETICS_TRAIN_DATA_DIR)
    init_cache(KINETICS_TEST_DATA_DIR)
    init_cache(KINETICS_VAL_DATA_DIR)

    # train_datapoints = populate_datapoints(KINETICS_TRAIN_CSV, split="train")
    # train_file = open(PICKLE_TRAIN_FILE, "wb+")
    # pickle.dump(train_datapoints, train_file)

    # test_datapoints = populate_datapoints(KINETICS_TEST_CSV, split="test")
    # test_file = open(PICKLE_TEST_FILE, "wb+")
    # pickle.dump(test_datapoints, test_file)

    val_datapoints = populate_datapoints(KINETICS_VAL_CSV, split="val")
    val_datapoints = clean_datapoints(val_datapoints)
    val_file = open(PICKLE_VAL_FILE, "wb+")
    pickle.dump(val_datapoints, val_file)
