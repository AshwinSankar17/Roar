__all__ = [
    'deprecated',
]

import functools
import inspect

import wrapt

from roar.utils import logging

# Remember which deprecation warnings have been printed already.
_PRINTED_WARNING = {}


def deprecated(wrapped=None, version=None, explanation=None):
    """
        Decorator which can be used for indicating that a function/class is deprecated and going to be removed.
        Tracks down which function/class printed the warning and will print it only once per call.

        Args:
            version: Version in which the function/class will be removed (optional).
            explanation: Additional explanation, e.g. "Please, ``use another_function`` instead." (optional).
    """

    if wrapped is None:
        return functools.partial(deprecated, version=version, explanation=explanation)

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        # Check if we already warned about that function.
        if wrapped.__name__ not in _PRINTED_WARNING.keys():
            # Add to list so we won't print it again.
            _PRINTED_WARNING[wrapped.__name__] = True

            # Prepare the warning message.
            entity_name = "Class" if inspect.isclass(wrapped) else "Function"
            msg = f"{entity_name} ``{wrapped.__name__}`` is deprecated."

            # Optionally, add version and explanation.
            if version is not None:
                msg = f"{msg} It is going to be removed in the {version} version."

            if explanation is not None:
                msg = f"{msg} {explanation}"

            # Display the deprecated warning.
            logging.warning(msg)

        # Call the function.
        return wrapped(*args, **kwargs)

    return wrapper(wrapped)
