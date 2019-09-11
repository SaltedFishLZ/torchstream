import collections
from torchstream.datasets.kinetics400 import Kinetics400


def test_kinetics_400():
    dataset_path = "/dnn/data/Kinetics/Kinetics-400-mp4"
    dataset = Kinetics400(root=dataset_path,
                          transform=Compose([Resize(256),
                                             CenterCrop(224),
                                             CenterSegment(32)]),
                          train=False, ext="mp4")
    print(dataset.__len__())

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=8,
                                             num_workers=32,
                                             pin_memory=True)

    num_samples_per_class = collections.OrderedDict()
    for vid, cid in tqdm.tqdm(dataloader):
        if cid in num_samples_per_class:
            num_samples_per_class[cid] += 1
        else:
            num_samples_per_class[cid] = 1
    print(num_samples_per_class)

if __name__ == "__main__":
    test_kinetics_400()
