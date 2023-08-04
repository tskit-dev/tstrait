import fractions

import numpy as np
import pytest

from tstrait.base import (
    _check_int,
    _check_val,
    _check_instance,
    _check_numeric_array,
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
        assert type(n) == float

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
