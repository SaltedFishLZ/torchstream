import os
import pickle
import csv
import logging
# import torch 
import torchstream.io.backends.opencv as backend
import torchstream.io.datapoint as datapoint
import numpy as np
import cv2

LOGGER_LEVEL = logging.WARN
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)

TRAIN_LABEL_FILE = "/dnn/data/Charades/Charades-v1-meta/Charades/Charades_v1_train.csv"
TEST_LABEL_FILE = "/dnn/data/Charades/Charades-v1-meta/Charades/Charades_v1_test.csv"

DATA_DIR = "/dnn/data/Charades"
DATA_MP4_DIR = os.path.join(DATA_DIR, "Charades-v1-mp4")
DATA_CLIPS_DIR = os.path.join(DATA_DIR, "Charades-v1-mp4-clips")

CHARADES_PICKLE_DIR = "/rscratch/bernie/video-acc/download/torchstream/datasets/charades"
PICKLE_TRAIN_FILE = os.path.join(CHARADES_PICKLE_DIR, "charades_training_split1.pkl")
PICKE_TEST_FILE = os.path.join(CHARADES_PICKLE_DIR, "charades_testing_split1.pkl")
cache = set()

def cls2int(x):
    return int(x[1:])

def convert_farray_color(farray, cin, cout, **kwargs):
    """
    Args:
        farray : input frame as a Numpy ndarray
        cin : input frame's color space
        cout : output frame's color space
        return value : output frame as a Numpy ndarray
    """
    if (cin == cout):
        return(farray)
    if (cin, cout) == ("BGR", "GRAY"):
        output = cv2.cvtColor(farray, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    elif (cin, cout) == ("BGR", "RGB"):
        output = cv2.cvtColor(farray, cv2.COLOR_BGR2RGB)
    elif (cin, cout) == ("RGB", "GRAY"):
        output = cv2.cvtColor(farray, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    elif (cin, cout) == ("RGB", "BGR"):
        output = cv2.cvtColor(farray, cv2.COLOR_RGB2BGR)
    elif (cin, cout) == ("GRAY", "BGR"):
        output = farray[:, :, 0]
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    elif (cin, cout) == ("GRAY", "RGB"):
        output = farray[:, :, 0]
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    else:
        assert NotImplementedError

    return(output)

def ndarray2video(varray, dst_path, cin="RGB", cout="BGR", fps=12, **kwargs):
    """Write a 4d array to a video file
    """
    t, h, w, c = varray.shape
    assert c == 3, NotImplementedError("Only accept color video now")

    writer = cv2.VideoWriter(dst_path,
                             cv2.VideoWriter_fourcc(*'MP4V'),
                             fps, (w, h))

    success = True
    for i in range(t):
        farray = varray[i, :, :, :]
        farray = convert_farray_color(farray, cin, cout)
        writer.write(farray)
    writer.release()

    # always successful
    return success

def generate_clips(labels):
    n = len(labels.keys())
    for i, (vid, actions) in enumerate(labels.items()):
        print("video {} ({} of {}) ########".format(vid, i, n))
        if vid in cache:
            continue
        cache.add(vid)

        src_path = os.path.join(DATA_MP4_DIR, "{}.mp4".format(vid))
        cap = cv2.VideoCapture(src_path)
        if (not cap.isOpened()):
            warn_str = "[video2ndarray] cannot open video {} \
                via cv2.VideoCapture ".format(video)
            logger.warn(warn_str)
            cap.release()
            continue
        f_fps = int(cap.get(cv2.CAP_PROP_FPS))
        # print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        f_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        varray = backend.video2ndarray(src_path)
        # print(varray.shape)
        
        for action in actions:
            start_frame = int(action['start'] * f_fps)
            end_frame = min(f_n-1, int(action['end'] * f_fps))
            if end_frame <= start_frame:
                logger.error("cannot generate clip for vid {}".format(vid))
                continue
            clip = varray[start_frame:end_frame+1]
            T1 = int(1000*action['start'])
            T2 = int(1000*action['end'])
            dst_path = os.path.join(DATA_CLIPS_DIR, "{}-{}-{}.mp4".format(vid, action['start'], action['end']))
            ndarray2video(clip, dst_path, fps=f_fps)
            # print(clip.shape)
        

def populate_datapoints(label_file):
    labels = parse_charades_csv(label_file)

    # print(labels)

    if not (os.path.exists(DATA_CLIPS_DIR) and
            os.path.isdir(DATA_CLIPS_DIR)):
        os.makedirs(DATA_CLIPS_DIR, exist_ok=True)

    generate_clips(labels)

    datapoints = []
    for filename in os.listdir(DATA_CLIPS_DIR):
        vid = filename.split('.')[0]
        start = vid.split('-')[1]
        end = vid.split('-')[2]
        label = filter(lambda x: x['start'] == start and x['end'] == end, 
                       labels[vid.split('-')[0]])[0]
        datapoints.append(Datapoint(DATA_DIR, 
                                    "Charades-v1-mp4-clips", 
                                    vid, 
                                    ext="mp4", 
                                    label=label))

    print(datapoints[0])
    return datapoints


def parse_charades_csv(filename):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{'class': x, 'start': float(
                    y), 'end': float(z)} for x, y, z in actions]
            labels[vid] = actions
    return labels

def init_cache():
    for filename in os.listdir(DATA_CLIPS_DIR):
        cache.add(filename.split('-')[0])

def main():
    init_cache()
    # print(cache)
    train_datapoints = populate_datapoints(TRAIN_LABEL_FILE)
    pickle.dump(train_datapoints, PICKLE_TRAIN_FILE)
    # test_datapoints = populate_datapoints(TEST_LABEL_FILE)
    # pickle.dump(test_datapoints, PICKLE_TEST_FILE)


if __name__ == "__main__":
    main()