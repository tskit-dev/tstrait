import fractions

import numpy as np
import pandas as pd
import pytest

from tstrait.base import (
    _check_int,
    _check_val,
    _check_instance,
    _check_numeric_array,
    _check_dataframe,
    _check_non_decreasing,
)  # noreorder


class TestInstance:
    def test_instance(self):
        n = _check_instance(1, "n", int)
        assert n == 1

    @pytest.mark.parametrize("n", [4.0, np.array([4]), fractions.Fraction(4, 1)])
    def test_instance_bad(self, n):
        with pytest.raises(TypeError, match=f"n must be a {int} instance"):
            _check_instance(n, "n", int)


class TestVal:
    @pytest.mark.parametrize(
        "n", [1.1, np.float16(1.1), np.int16(1), np.array([1.1])[0]]
    )
    def test_val(self, n):
        n = _check_val(n, "n")
        assert isinstance(n, float)

    @pytest.mark.parametrize("n", [4j, np.array([4]), "1"])
    def test_val_bad(self, n):
        with pytest.raises(TypeError, match="n must be numeric"):
            _check_val(n, "n")

    @pytest.mark.parametrize("n", [0, -1])
    def test_val_below_min(self, n):
        with pytest.raises(ValueError, match="n must be a number greater than 0"):
            _check_val(n, "n", 0)

    def test_val_inclusive(self):
        _check_val(0, "n", 0, inclusive=True)

    def test_val_below_min_inclusive(self):
        with pytest.raises(ValueError, match="n must be a number not " "less than 0"):
            _check_val(-1, "n", 0, inclusive=True)


class TestInt:
    @pytest.mark.parametrize("n", [1, np.uint8(1), np.int16(1), np.array(1)])
    def test_int(self, n):
        n = _check_int(n, "n")
        assert n == 1

    @pytest.mark.parametrize("n", [4.0, np.array([4]), fractions.Fraction(4, 1)])
    def test_int_bad(self, n):
        with pytest.raises(TypeError, match="n must be an integer"):
            _check_int(n, "n")

    def test_int_below_min(self):
        with pytest.raises(ValueError, match="n must be an integer not " "less than 0"):
            _check_int(-1, "n", 0)


class TestNumericArray:
    def test_true(self):
        _check_numeric_array([1, 2.1], "input")
        _check_numeric_array(np.array([1, 2.1]), "input")

    def test_nonnumeric(self):
        with pytest.raises(TypeError, match="input must be numeric"):
            _check_numeric_array([1, "1"], "input")


class TestDataFrame:
    def test_type(self):
        with pytest.raises(
            TypeError,
            match="df must be a <class 'pandas.core.frame.DataFrame'> instance",
        ):
            _check_dataframe(1, {"one"}, "df")

    def test_dataframe(self):
        df = pd.DataFrame({"first": [0, 1], "second": [3, 4]})

        with pytest.raises(
            ValueError, match="columns must be included in df dataframe"
        ):
            _check_dataframe(df, ["column"], "df")

        pd.testing.assert_frame_equal(
            _check_dataframe(df, ["first"], "df"), df[["first"]]
        )

        pd.testing.assert_frame_equal(
            _check_dataframe(df, ["first", "second"], "df"), df
        )


class TestNonDecreasing:
    def test_non_decreasing(self):
        with pytest.raises(ValueError, match="array must be non-decreasing"):
            _check_non_decreasing([0, 1, 0], "array")

        _check_non_decreasing([0, 0, 1, 1, 4], "array")
        _check_non_decreasing([0], "array")
