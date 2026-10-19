def check_slice(inp: slice, upper_bound: int):
    if 0 > inp.start >= upper_bound:
        raise ValueError("slice start out of bounds")
    elif 0 > inp.stop >= upper_bound:
        raise ValueError("slice stop out of bounds")
    elif inp.step == 0:
        raise ValueError("slice step must not be zero")
    elif inp.step > 0 and inp.stop < inp.start:
        raise ValueError("slice step must be negative if stop is before start")
    elif inp.step < 0 and inp.start < inp.stop:
        raise ValueError("slice step must be positive if stop is before start")