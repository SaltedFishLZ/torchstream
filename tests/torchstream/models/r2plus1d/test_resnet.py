import torch
from torchstream.models.r2plus1d import r2plus1d_18


def test_r2plus1d_18():
    model = r2plus1d_18(pretrained=True)

    for _t in [8, 16]:
        for _s in range(32, 112, 16):
            print(_t, _s, _s)
            volume = torch.randn(1, 3, _t, _s, _s)
            output = model(volume)


if __name__ == "__main__":
    test_r2plus1d_18()