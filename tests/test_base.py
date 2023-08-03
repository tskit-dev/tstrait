import fractions

import numpy as np
import pytest

from tstrait.base import (
    _define_rng,
    _check_int,
    _check_val,
    _check_instance,
    _check_symmetric,
    _num_samples,
    _check_same_length,
    _is_arraylike,
    _check_numeric_array,
)  # noreorder


class TestRNG:
    def test_rng(self):
        rng5 = np.random.default_rng(5)
        np.testing.assert_allclose(_define_rng(5).choice(100, 10), rng5.choice(100, 10))
        assert isinstance(_define_rng(rng5), np.random.Generator)

        rng10 = np.random.default_rng(10)
        assert not np.array_equal(_define_rng(5).choice(100, 10), rng10.choice(100, 10))
        seed = "invalid"
        with pytest.raises(
            ValueError,
            match=f"{seed!r} cannot be used to construct a new random generator with "
            "numpy.random.default_rng",
        ):
            _define_rng(seed)


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


class TestSymmetric:
    def test_symmetric(self):
        arr_sym = np.array([[0, 1], [1, 2]])
        arr_bad = np.ones(2)
        arr_asym = np.array([[0, 2], [0, 2]])
        arr_str = np.array([["1", "a"], [1, 2]])

        with pytest.raises(ValueError, match="array must be 2-dimensional and square"):
            _check_symmetric(arr_bad, "array")

        with pytest.raises(ValueError, match="array must be symmetric"):
            _check_symmetric(arr_asym, "array")

        with pytest.raises(TypeError, match="array must be numeric"):
            _check_symmetric(arr_str, "array")

        with pytest.raises(TypeError, match="array must be array-like"):
            _check_symmetric(1, "array")

        np.testing.assert_allclose(arr_sym, _check_symmetric(arr_sym, "array"))


class TestNumSample:
    def test_retrieve_samples_from_non_standard_shape(self):
        class TestNonNumericShape:
            def __init__(self):
                self.shape = ("not numeric",)

            def __len__(self):
                return len([1, 2, 3])

        X = TestNonNumericShape()
        assert _num_samples(X) == len(X)

        # check that it gives a good error if there's no __len__
        class TestNoLenWeirdShape:
            def __init__(self):
                self.shape = ("not numeric",)

        with pytest.raises(TypeError, match="Expected sequence or array-like"):
            _num_samples(TestNoLenWeirdShape())


class TestSameLength:
    def test_check_consistent_length(self):
        _check_same_length([1], [2], "one", "two")
        _check_same_length([[1, 2], [[1, 2]]], [1, 2], "one", "two")
        with pytest.raises(
            ValueError,
            match="Input variables in one and two have inconsistent dimensions",
        ):
            _check_same_length([1, 2], [2], "one", "two")
        with pytest.raises(
            TypeError, match="Expected sequence or array-like, got <class 'int'>"
        ):
            _check_same_length([1, 2], 1, "one", "two")
        with pytest.raises(
            TypeError, match="Expected sequence or array-like, got <class 'object'>"
        ):
            _check_same_length([1, 2], object(), "one", "two")
        with pytest.raises(TypeError, match="Singleton array..."):
            _check_same_length([1, 2], np.array(1), "one", "two")


class TestArrayLike:
    def test_is_arraylike(self):
        assert _is_arraylike([1]) is True
        assert _is_arraylike(np.array([1, 2])) is True
        assert _is_arraylike(1) is False


class TestNumericArray:
    def test_true(self):
        _check_numeric_array([1, 2.1], "input")
        _check_numeric_array(np.array([1, 2.1]), "input")

    def test_nonarray(self):
        with pytest.raises(TypeError, match="input must be array-like"):
            _check_numeric_array(1, "input")

    def test_nonnumeric(self):
        with pytest.raises(TypeError, match="input must be numeric"):
            _check_numeric_array([1, "1"], "input")
