
if __name__ == "__main__":
    DST_SEQ_SAMPLE = DataPoint(root=DIR_PATH,
                               rpath="test_seq", name="test_seq", ext="jpg")
    print(vid2seq(DST_VID_SAMPLE, DST_SEQ_SAMPLE))