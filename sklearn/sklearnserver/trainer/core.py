from functools import partial
from functools import wraps
import numpy as np


# def _preprocess(inputs):
#     return inputs


# def add_preprocessing(prep_func):
#     def actual_decorator(func):
#         @wraps(func)
#         def wrapper(inputs):
#             prep_func(*args, **kwargs)
#             return func(*args, **kwargs)
#         return wrapper
#     return actual_decorator


# def preprocess(func):
#     @wraps(func)
#     def profiler(*args, **kwargs):
#         return prep_func(*args, **kwargs)

#     return profiler

class BaseModel(object):
    def __init__(self,  filepath=None, *args, **kwargs):
        super().__init__()
        if filepath:
            self.load(filepath)
        else:
            try:
                self.build(*args, **kwargs)
            except TypeError as e:
                raise e

    def preprocess(self, X):
        return X

    def postprocess(self, y_hat):
        return y_hat

    def build(self, *args, **kwargs):
        pass

    def train(self, X, y, *args, **kwargs):
        pass

    def evaluate(self, X, y, *args, **kwargs):
        pass

    def predict(self, X, *args, **kwargs):
        pass

    def save(self, filepath, *args, **kwargs):
        pass

    def load(self, filepath, *args, **kwargs):
        pass


