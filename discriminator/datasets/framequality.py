"""
"""
import os
import pickle

import torch
import torch.utils.data as data

from torchstream.datasets import SomethingSomethingV1

# ------------------------------------------------------------------------- #
#                   Main Classes (To Be Used outside)                       #
# ------------------------------------------------------------------------- #

## frame quality for video recognition
class FrameQualityDataset(data.Dataset):
    """Frame Quality Dataset
    """
    def __init__(self, trace_root, chances=50, num_snippets=8, num_frames=16,
                 train=True, transform=None, target_transform=None,
                 **kwargs):
        """
        Args:

        """
        check_str = "s{}.f{}.{}".format(num_snippets, num_frames,
                                        "train" if train else "test")
        assert check_str in trace_root, ValueError("Invalid trace toor")

        self.trace_root = trace_root
        self.chances = chances
        self.num_snippets = num_snippets
        self.num_frames = num_frames

        self.video_dataset = SomethingSomethingV1(
            train=train,
            transform=transform,
            target_transform=target_transform,
            **kwargs
            )

        self.indices = None
        self.corrects = None
        for chance in range(chances):

            chance_dir_path = os.path.join(trace_root, "chance{}".format(chance))
            index_file_path = os.path.join(chance_dir_path, "index.pkl")
            correct_file_path = os.path.join(chance_dir_path, "correct.pkl")

            with open(index_file_path, "rb") as f:
                index = pickle.load(f)
                index.unsqueeze_(dim=1)
                if self.indices is None:
                    self.indices = index
                else:
                    self.indices = torch.cat((self.indices, index), dim=1)

            with open(correct_file_path, "rb") as f:
                correct = pickle.load(f)
                correct.unsqueeze_(dim=1)
                if self.corrects is None:
                    self.corrects = correct
                else:
                    self.corrects = torch.cat((self.corrects, correct), dim=1)

        # filter samples
        selected_video_indices = []
        for i in range(len(self.video_dataset)):
            correct_num = int(self.corrects[i].sum())
            # print(correct_num)
            correct_ratio = correct_num / self.chances
            if (correct_ratio > 0.5) and (correct_ratio < 0.9):
                selected_video_indices.append(i)
        # print(len(selected_video_indices))
        new_datapoints = [self.video_dataset.datapoints[_i] for _i in selected_video_indices]
        self.video_dataset.datapoints = new_datapoints
        new_samples = [self.video_dataset.samples[_i] for _i in selected_video_indices]
        self.video_dataset.samples = new_samples
        self.corrects = self.corrects[selected_video_indices]
        self.indices = self.indices[selected_video_indices]

    def __len__(self):
        return len(self.video_dataset) * self.chances

    def __getitem__(self, idx):
        """
            [video][chance]
            (index, blob, cid)
        """
        video_id = idx // self.chances
        chance_id = idx % self.chances

        blob, _ = self.video_dataset[video_id]

        index = self.indices[video_id][chance_id]
        index_onehot = torch.zeros(self.num_frames)
        index_onehot.scatter_(0, index.long(), 1)

        cid = self.corrects[video_id][chance_id]

        return (blob, index_onehot, cid)


if __name__ == "__main__":
    dataset = FrameQualityDataset(trace_root="../kfs/mc-traces/s8.f16.train")
    print(dataset.indices.size())
    print(dataset.corrects.size())

    print(len(dataset))

    import tqdm
    for i in (range(len(dataset))):
        blob, index, cid = dataset[i]
        print(blob.size(), index, cid)
