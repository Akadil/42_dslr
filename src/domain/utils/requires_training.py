import functools


def requires_training(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self.is_trained()  # raises if not trained
        return method(self, *args, **kwargs)

    return wrapper
