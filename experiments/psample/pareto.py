"""
"""

def pareto_dominate(x, y):
    if (
            (x["flops"] <= y["flops"])
            and (x["accuracy"] >= y["accuracy"])
    ):
        return True
    return False

def update_pareto_list(plist, x):
    assert isinstance(plist, list), TypeError
    if not plist:
        plist.append(x)
        return plist

    for y in plist:
        if pareto_dominate(x, y):
            plist.remove(y)

    flag = True
    for y in plist:
        if pareto_dominate(y, x):
            flag = False
            break
    if flag:
        plist.append(x)

    return plist
