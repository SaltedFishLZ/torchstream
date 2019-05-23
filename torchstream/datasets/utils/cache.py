import hashlib

LINUX_RESTRICTED_CHARS = ['/', '>', '<', '|', ':', '&', '*']

def sorted_dictitems(kwargs):
    _pairs = sorted(kwargs.items())
    for _i in range(len(_pairs)):
        _key, _val = _pairs[_i]
        if isinstance(_val, dict):
            _val = sorted_dictitems(_val)
        _pairs[_i] = _key, _val
    return _pairs

def hashid(x):
    if isinstance(x, dict):
        _pairs = sorted_dictitems(x)
        string = str(_pairs)
    else:
        string = str(_pairs)
    hasher = hashlib.md5(string.encode(encoding="utf-8"))
    hashid = hasher.hexdigest()
    return hashid

def hashstr(depth=0, maxlen=254, **kwargs):
    string = ""
    if depth <= 0:
        return str(hashid(kwargs))
    ## recurevisely generate
    for key in kwargs:
        val = kwargs[key]
        if isinstance(val, dict):
            _s = hashstr(depth=depth-1, maxlen=maxlen, **val)
            print("depth", depth, _s)
        else:
            _s = str(val)
        _s = '_'.join([str(key), _s])
        for _c in LINUX_RESTRICTED_CHARS:
            _s = _s.replace(_c, "_")
        string += "." + _s
    return string[0:maxlen]

