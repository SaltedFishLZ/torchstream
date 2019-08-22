from torchstream.utils.mapreduce import Worker, Manager

def square_sum_test():
    r"""
    Calculate \Sigma i^2
    """
    N = 1000000
    REF = (N-1)*N*(2*N-1) // 6

    tasks = []
    for _i in range(N):
        tasks.append({"x": _i})

    def mapper(x):
        return(x**2)

    def reducer(inputs):
        sum = 0
        for _x in inputs:
            sum += _x 
        return [sum]

    manager = Manager(name="self-test", mapper=mapper, reducer=reducer)
    manager.hire(worker_num=16)
    result = manager.launch(tasks=tasks, progress=True)
    
    if (len(result) == 1) and (REF == result[0]):
        print("[Square Sum Test] Passed")
        return True
    else:
        print(
            "[Square Sum Test] Failed, {} expected, {} got".format(REF, result))
        return False


if __name__ == "__main__":
    ret = square_sum_test()
    assert ret
