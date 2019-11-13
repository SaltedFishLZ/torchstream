from torchstream.models.r2plus1d import r2plus1d_18


def test_r2plus1d_18():
    model = r2plus1d_18(pretrained=True)
    print(model)
    print(model.state_dict().keys())


if __name__ == "__main__":
    test_r2plus1d_18()