"""
Author: Bernie Wang
"""
import os
import pickle
import csv

import torchstream.io.backends.opencv as backend
from torchstream.io.datapoint import DataPoint, UNKNOWN_LABEL
from torchstream.utils.download import download
from torchstream.utils.mapreduce import Manager

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

# metadata source links
DOWNLOAD_SERVER_PREFIX = ("zhen@a18.millennium.berkeley.edu:"
                          "/home/eecs/zhen/video-acc/download/")
DOWNLOAD_SRC_DIR = "tools/datasets/metadata/kinetics400"
KINETICS_TRAIN_CSV_SRC = os.path.join(DOWNLOAD_SRC_DIR,
                                      "kinetics-400_train.csv")
KINETICS_TEST_CSV_SRC = os.path.join(DOWNLOAD_SRC_DIR,
                                     "kinetics-400_test.csv")
KINETICS_VAL_CSV_SRC = os.path.join(DOWNLOAD_SRC_DIR,
                                    "kinetics-400_val.csv")

# local metadata
KINETICS_TRAIN_CSV_PATH = os.path.join(DOWNLOAD_SRC_DIR,
                                       "kinetics-400_train.csv")
KINETICS_TEST_CSV_PATH = os.path.join(DOWNLOAD_SRC_DIR,
                                      "kinetics-400_test.csv")
KINETICS_VAL_CSV_PATH = os.path.join(DOWNLOAD_SRC_DIR,
                                     "kinetics-400_val.csv")
# download metadata when missing
if not os.path.exists(KINETICS_TRAIN_CSV_PATH):
    download(KINETICS_TRAIN_CSV_SRC, KINETICS_TRAIN_CSV_PATH)
if not os.path.exists(KINETICS_TEST_CSV_PATH):
    download(KINETICS_TEST_CSV_SRC, KINETICS_TEST_CSV_PATH)
if not os.path.exists(KINETICS_VAL_CSV_PATH):
    download(KINETICS_VAL_CSV_SRC, KINETICS_VAL_CSV_PATH)

# dataset links
KINETICS_DATA_DIR = os.path.expanduser("~/Datasets/Kinetics/Kinetics-400-mp4")
KINETICS_TRAIN_DATA_DIR = os.path.join(KINETICS_DATA_DIR, "train")
KINETICS_TEST_DATA_DIR = os.path.join(KINETICS_DATA_DIR, "test")
KINETICS_VAL_DATA_DIR = os.path.join(KINETICS_DATA_DIR, "val")

# destination links
KINETICS_LABEL_FILE = os.path.join(DIR_PATH, "kinetics400_labels.txt")
KINETICS_PICKLE_DIR = DIR_PATH
PICKLE_TRAIN_FILE = os.path.join(KINETICS_PICKLE_DIR, "kinetics400_mp4_training.pkl")
PICKLE_TEST_FILE = os.path.join(KINETICS_PICKLE_DIR, "kinetics400_mp4_testing.pkl")
PICKLE_VAL_FILE = os.path.join(KINETICS_PICKLE_DIR, "kinetics400_mp4_validation.pkl")


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

    print(corruptpoints)
    corruptnames = [dp.name for dp in corruptpoints]

    cleanpoints = []
    for idx, dp in enumerate(datapoints):
        if dp.name in corruptnames:
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

    # train_datapoints = populate_datapoints(KINETICS_TRAIN_CSV_PATH, split="train")
    # train_file = open(PICKLE_TRAIN_FILE, "wb+")
    # pickle.dump(train_datapoints, train_file)

    # test_datapoints = populate_datapoints(KINETICS_TEST_CSV_PATH, split="test")
    # test_file = open(PICKLE_TEST_FILE, "wb+")
    # pickle.dump(test_datapoints, test_file)

    val_datapoints = populate_datapoints(KINETICS_VAL_CSV_PATH, split="val")
    val_datapoints = clean_datapoints(val_datapoints)
    val_file = open(PICKLE_VAL_FILE, "wb+")
    pickle.dump(val_datapoints, val_file)
