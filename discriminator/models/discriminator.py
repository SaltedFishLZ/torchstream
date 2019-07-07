import sys
import collections

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

import torch
import torch.nn as nn

from torchstream.models import TSM
from torchstream.ops import Identity

class FrameQualityDiscriminator(nn.Module):
    """
    """
    def __init__(self, input_size):
        """
        """
        assert isinstance(input_size, Sequence), TypeError
        assert len(input_size) == 3, ValueError
        super(FrameQualityDiscriminator, self).__init__()
        self.input_size = input_size

        self.video_feature_extractor = TSM(cls_num=174,
                                           input_size=input_size)
        num_image_features = self.video_feature_extractor.base_model.fc.fc.in_features
        num_frames = self.video_feature_extractor.input_size[0]
        num_video_features = num_image_features * num_frames

        self.video_feature_extractor.base_model.fc = Identity()
        self.video_feature_extractor.consensus = Identity()

        num_index_features = num_frames * num_video_features
        self.index_feature_extractor = nn.Linear(
            in_features=num_frames,
            out_features=num_index_features
            )

        self.aggregation_network = nn.Sequential(
            nn.Linear(
                in_features=num_video_features + num_index_features,
                out_features=1024
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=1024,
                out_features=256
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=2
            ),
        )

    @property
    def num_frames(self):
        return self.input_size[0]

    def forward(self, x):
        assert isinstance(x, Sequence), TypeError
        assert len(x) == 2, ValueError
        
        video, index = x
        
        v_f = self.video_feature_extractor(video)
        i_f = self.index_feature_extractor(index)
        
        feature = torch.cat((v_f, i_f), dim=-1)

        out = self.aggregation_network(feature)

        return out


if __name__ == "__main__":
    discriminator = FrameQualityDiscriminator(input_size=[16, 224, 224])
    vid = torch.rand(1, 16, 224, 224)
    idx = torch.round(torch.rand(1, 16))
    print(idx)