import numbers
import operator

import numpy as np


def _define_rng(seed):
    """Generater a numpy.random.Generator instance from the seed input."""
    if seed is None:
        return np.random.default_rng(None)
    if isinstance(seed, int) and seed >= 0:
        return np.random.default_rng(seed)
    if (
        isinstance(seed, np.random.SeedSequence)
        or isinstance(seed, np.random.BitGenerator)
        or isinstance(seed, np.random.Generator)
    ):
        return np.random.default_rng(seed)
    raise ValueError(
        f"{seed!r} cannot be used to construct a new random generator with "
        "numpy.random.default_rng"
    )


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


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered a valid input" % x)
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def _check_same_length(array1, array2, name1, name2):
    if _num_samples(array1) != _num_samples(array2):
        raise ValueError(
            f"Input variables in {name1} and {name2} have inconsistent dimensions"
        )


def _is_arraylike(x):
    """Returns whether the input is array-like."""
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _check_numeric_array(array, name):
    """Check if the input is a numeric array or not"""
    if not _is_arraylike(array):
        raise TypeError(f"{name} must be array-like")
    result = np.array([_check_val(value, name) for value in array])
    return result


def _check_symmetric(array, name):
    """Make sure that the input array is 2D, square and symmetric."""
    if not _is_arraylike(array):
        raise TypeError(f"{name} must be array-like")
    array = np.array(array)
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError(f"{name} must be 2-dimensional and square")
    try:
        symmetric = np.allclose(array, array.T, atol=1e-08)
    except TypeError as type_error:
        raise TypeError(f"{name} must be numeric") from type_error

    if not symmetric:
        raise ValueError(f"{name} must be symmetric")

    return array
