"""
"""
import copy

from torchstream.io.datapoint import DataPoint


def test_datapoint():
    a = DataPoint(root="Foo", reldir="Bar", name="test", ext="avi")
    # print(a)

    b = copy.deepcopy(a)
    b.root = "Fooood"
    # print(b)

    c = copy.deepcopy(b)
    c.ext = "mp4"
    # print(c)

    assert not (a == c)
    assert not (a < c)
    assert not (a > c)

    d = DataPoint(root="Foo", reldir="Bar", name="aha", ext="avi")
    assert (d < a)

    d1000 = DataPoint(root="Foo", reldir="Bar", name="1000", ext="avi")
    d9 = DataPoint(root="Foo", reldir="Bar", name="9", ext="avi")
    assert (d9 < d1000)
    assert not (d9 > d1000)


if __name__ == "__main__":
    test_datapoint()
