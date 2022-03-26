import os
import time
from functools import wraps
from contextlib import contextmanager

DEBUG = bool(os.getenv("debug"))


def time_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {:.3f}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


@contextmanager
def time_block(label):
    if DEBUG:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            print('{} : {:.3f}'.format(label, end - start))
    else:
        try:
            yield
        finally:
            pass


class Timer:
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
