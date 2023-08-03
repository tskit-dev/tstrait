import numbers
import operator

import numpy as np


def _check_instance(data, name, input_type):
    """Check if the input is from the specified instance."""
    if not isinstance(data, input_type):
        raise TypeError(f"{name} must be a {input_type} instance")
    return data


def _check_val(value, name, minimum=None, inclusive=False):
    """Make sure that the input is numerical and greater than the minimum value."""
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if minimum is not None and not inclusive and value <= minimum:
        raise ValueError(f"{name} must be a number greater than {minimum}")
    elif minimum is not None and inclusive and value < minimum:
        raise ValueError(f"{name} must be a number not less than {minimum}")
    return value


def _check_int(value, name, minimum=None):
    """Make sure that the input is an integer and greater than the minimum value."""
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be an integer not less " f"than {minimum}"
        ) from None
    return value


def _check_numeric_array(array, name):
    """Check if the input is a numeric array or not. This is used in unit tests."""
    result = np.array([_check_val(value, name) for value in array])
    return result
