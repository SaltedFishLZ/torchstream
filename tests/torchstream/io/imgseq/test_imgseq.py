

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


def TestImageSequence():
    test_video = os.path.join(DIR_PATH, "test.avi")
    test_frames = os.path.join(DIR_PATH, "test_frames")
    from .utils.vision import video2frames, farray_show
    video2frames(test_video, test_frames)

    imgseq_0 = ImageSequence(path=test_frames,
                             ext="jpg", cin="BGR", cout="RGB"
                             )

    varray = imgseq_0.get_varray()
    print(varray.shape)
    # print(imgseq_0.get_farray(0).shape)
    # farray_show(caption="test", farray=farray)

    # import cv2
    # (cv2.waitKey(0) & 0xFF == ord("q"))
    # cv2.destroyAllWindows()

    import importlib

    dataset = "weizmann"
    metaset = importlib.import_module(
        "datasets.metadata.metasets.{}".format(dataset))

    kwargs = {
        "root" : metaset.JPG_DATA_PATH,
        "layout" : metaset.__layout__,
        "lbls" : metaset.__LABELS__,
        "mod" : "RGB",
        "ext" : "jpg",
    }
    
    from .metadata.collect import collect_datapoints
    datapoints = collect_datapoints(**kwargs)

    for datapoint in datapoints:
        imgseq = ImageSequence(datapoint)
        print(np.all(np.array(imgseq) == imgseq.get_varray()))


if __name__ == "__main__":
    TestImageSequence()
