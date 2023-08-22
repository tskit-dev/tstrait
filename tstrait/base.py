import numbers
import operator

import numpy as np
import pandas as pd


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


def _check_dataframe(df, column, df_name):
    """Check if `df` is a pandas dataframe or not and examine if `column` is a column
    in `df` or not
    column is the set of column names and df_name is the name of df
    """

    df = _check_instance(df, df_name, pd.DataFrame)
    if not set(column).issubset(df.columns):
        raise ValueError(f"{column} columns must be included in {df_name} dataframe")
    return df[list(column)]


def _check_non_decreasing(array, array_name):
    """Check that the array is non-decreasing."""
    if len(array) > 1:
        diff = np.diff(array)
        if np.min(diff) < 0:
            raise ValueError(f"{array_name} must be non-decreasing")
