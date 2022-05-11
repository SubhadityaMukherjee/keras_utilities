def split_by_func(items, f):
    """
    Split a list by a function
    """
    mask = [f(o) for o in items]
    f = [o for o, m in zip(items, mask) if m == False]
    t = [o for o, m in zip(items, mask) if m == True]
    return f, t
