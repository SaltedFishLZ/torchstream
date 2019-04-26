
from .common import trainset_df, valset_df, testset_df

data_labels = dict()

for idx, row in trainset_df.iterrows():
    video = row["video"]
    label = row["label"]
    data_labels[video] = label

for idx, row in valset_df.iterrows():
    video = row["video"]
    label = row["label"]
    data_labels[video] = label

if __name__ == "__main__":
    print(len(data_labels))