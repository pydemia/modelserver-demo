from functools import partial
from functools import wraps



def _preprocess(inputs):
    return inputs


def add_preprocessing(prep_func):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(inputs):
            prep_func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return actual_decorator


def preprocess(func):
    @wraps(func)
    def profiler(*args, **kwargs):
        return prep_func(*args, **kwargs)

    return profiler
