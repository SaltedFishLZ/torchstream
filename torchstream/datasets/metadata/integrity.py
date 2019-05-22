"""
"""
__all__ = [
    "check_duplicated_sample", "check_label_matching",
    "check_sample_numer",
    "check_integrity"
]

import os
import logging

from . import __config__
from .sample import Sample, SampleSet

# ---------------------------------------------------------------- #
#                  Configuring Python Logger                       #
# ---------------------------------------------------------------- #

if __config__.__VERY_VERBOSE__:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s"
    )
elif __config__.__VERY_VERBOSE__:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s - %(levelname)s - %(message)s"
    )
elif __config__.__VERBOSE__:
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
else:
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
logger = logging.getLogger(__name__)


def check_duplicated_sample(samples):
    """ Check whether there is a duplicated sample
    """
    assert isinstance(samples, set), TypeError
    names = set()
    for _sample in samples:
        if _sample.name in names:
            err_str = "duplicated sample [{}]".format(_sample.name)
            logger.error(err_str)
            return False
        names.add(_sample.name)
    return True

def check_label_matching(samples, lbls):
    """Check whether labels are matching
    """
    assert isinstance(lbls, (dict, set)), TypeError
    assert isinstance(samples, set), TypeError

    if not isinstance(samples, SampleSet):
        lbls_got = SampleSet(samples).counts
    else:
        lbls_got = samples.counts

    if set(lbls) != set(lbls_got):
        err_str = "mismatching labels:"
        err_str += "Expected:\n{}\n".format(lbls)
        err_str += "Got:\n{}\n".format(lbls)
        logger.error(err_str)
        return False
    return True

def check_sample_numer(samples, lbls):
    """Check sample number is valid or not
    """
    assert isinstance(lbls, (dict, set)), TypeError
    assert isinstance(samples, set), TypeError

    if not isinstance(samples, SampleSet):
        lbls_got = SampleSet(samples).counts
    else:
        lbls_got = samples.counts

    passed = True
    for _label in lbls_got:
        _sample_count = lbls_got[_label]
        _reference_count = lbls[_label]
        assert isinstance(_reference_count, list), TypeError
        assert len(_reference_count) == 2, "Invalid Interval"
        lower = _reference_count[0]
        upper = _reference_count[1]
        assert lower <= upper, "Invalid Interval"
        if not (_sample_count >= lower and _sample_count <= upper):
            err_str = "[{}] sample number mismatch:\n"
            err_str += "should in {}, but got [{}]"
            err_str = err_str.format(_label, _sample_count, _reference_count)
            logger.error(err_str)
            passed = False

    return passed

def check_integrity(samples, lbls):
    """ Check metadata integrity
    """
    passed = True

    passed = passed and check_duplicated_sample(samples)
    passed = passed and check_label_matching(samples, lbls)
    passed = passed and check_sample_numer(samples, lbls)

    return True

def test():
    import importlib

    dataset = "sth_sth_v1"
    metaset = importlib.import_module(
        "datasets.metadata.metasets.{}".format(dataset))

    kwargs = {
        "root" : metaset.JPG_DATA_PATH,
        "layout" : metaset.__layout__,
        "annots" : metaset.__ANNOTATIONS__,
        "lbls" : metaset.__LABELS__,
        "mod" : "RGB",
        "ext" : "jpg",
    }

    from . collect import collect_samples
    samples = collect_samples(**kwargs)

    lbls = metaset.__SAMPLES_PER_LABEL__

    print(check_integrity(samples, lbls))


if __name__ == "__main__":
    test()