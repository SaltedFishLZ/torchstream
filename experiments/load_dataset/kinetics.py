import os
import pickle
import csv
import logging
# import torch 
import torchstream.io.backends.opencv as backend
from torchstream.io.datapoint import DataPoint
import numpy as np
import cv2

KINETICS_METADATA_DIR = '/dnn/data/ActivityNet/Crawler/Kinetics/data/'
KINETICS_TRAIN_CSV = os.path.join(KINETICS_METADATA_DIR, 'kinetics-400_train.csv')
KINETICS_VAL_CSV = os.path.join(KINETICS_METADATA_DIR, 'kinetics-400_val.csv')

DOWNLOAD_SERVER_PREFIX = os.path.expanduser('~/video-acc/download/torchstream/datasets/kinetics/')
KINETICS_LABEL_FILE = os.path.join(DOWNLOAD_SERVER_PREFIX, "kinetics-400_labels.txt")

KINETICS_DATA_DIR = '/dnn/data/Kinetics/Kinetics-400-mp4'
KINETICS_TRAIN_DATA_DIR = os.path.join(KINETICS_DATA_DIR, "train")
KINETICS_VAL_DATA_DIR = os.path.join(KINETICS_DATA_DIR, "val")

KINETICS_PICKLE_DIR = os.path.expanduser("~/video-acc/download/torchstream/datasets/kinetics")
PICKLE_TRAIN_FILE = os.path.join(KINETICS_PICKLE_DIR, "kinetics_training_split1.pkl")
PICKLE_VAL_FILE = os.path.join(KINETICS_PICKLE_DIR, "kinetics_val_split1.pkl")

TRAIN_CORRUPT_INDICES = [1139, 6112, 9486, 18162, 23131, 27903, 35247, 39514, 49851, 60177, 61523, 74810, 86195, 105111, 109340, 117082, 117257, 127920, 133607, 134597, 134660, 136602, 146561, 147639, 154674, 157595, 183509, 184997, 191015, 197140, 197402, 217717, 219969]
VAL_CORRUPT_INDICES = [12809, 14334]

vid2filename = {}

def write_classes():
    if not os.path.exists(KINETICS_LABEL_FILE):
        classes = []
        for label_class in os.listdir(KINETICS_TRAIN_DATA_DIR):
            classes.append(label_class)

        classes.sort()

        with open(KINETICS_LABEL_FILE, 'w') as f:
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
            if i not in corrupt_files:
                vid = row['youtube_id']
                label = row['label']
                labels.append((vid, label))
    return labels

def populate_datapoints(label_file, split=0, corrupt_files=[]):
    labels = parse_kinetics_csv(label_file, corrupt_files=corrupt_files)

    datapoints = []
    for vid, label in labels:
        if vid in vid2filename:
            if split == 0:
                rel_path = os.path.join("train", label)
            elif split == 1:
                rel_path = os.path.join("val", label)
            else:
                rel_path = os.path.join("test", "test")
            
            datapoints.append(DataPoint(KINETICS_DATA_DIR, 
                                        rel_path, 
                                        vid2filename[vid], 
                                        ext="mp4", 
                                        label=label))

    return datapoints


if __name__ == "__main__":
    write_classes()
    init_cache(KINETICS_TRAIN_DATA_DIR)
    init_cache(KINETICS_VAL_DATA_DIR)

    train_datapoints = populate_datapoints(KINETICS_TRAIN_CSV, split=0, corrupt_files=TRAIN_CORRUPT_INDICES)
    train_file = open(PICKLE_TRAIN_FILE, 'wb+')
    pickle.dump(train_datapoints, train_file)

    val_datapoints = populate_datapoints(KINETICS_VAL_CSV, split=1, corrupt_files=VAL_CORRUPT_INDICES)
    val_file = open(PICKLE_VAL_FILE, 'wb+')
    pickle.dump(val_datapoints, val_file)


