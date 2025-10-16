import time, functools


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        dur = time.perf_counter() - start
        print(f"[TIMER] {func.__name__} executed in {dur:.3f}s")
        return result

    return wrapper
