
def test():

    import importlib

    dataset = "weizmann"
    metaset = importlib.import_module(
        "datasets.metadata.metasets.{}".format(dataset))

    kwargs = {
        "root" : metaset.AVI_DATA_PATH,
        "layout" : metaset.__layout__,
        "lbls" : metaset.__LABELS__,
        "mod" : "RGB",
        "ext" : "avi",
    }

    from .metadata.collect import collect_samples
    samples = collect_samples(**kwargs)

    for _sample in samples:
        vid_arr = VideoArray(_sample, lazy=False)
        print(vid_arr)
    # print(np.array(vid_arr))

if __name__ == "__main__":
    test()