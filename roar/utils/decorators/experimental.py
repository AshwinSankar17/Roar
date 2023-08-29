__all__ = ['experimental']


import wrapt

from roar.utils import logging


@wrapt.decorator
def experimental(wrapped, instance, args, kwargs):
    logging.warning(f"`{wrapped}` is experimental and not ready for production yet. Use at your own risk.")
    return wrapped(*args, **kwargs)
