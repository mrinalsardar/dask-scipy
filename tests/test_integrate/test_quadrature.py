import dask.array as da
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from dask_scipy.integrate import newton_cotes, romb, simpson


def test_simpson():
    """_summary_"""
    y = da.arange(17)
    assert_equal(simpson(y).compute(), 128)
    assert_equal(simpson(y, dx=0.5).compute(), 64)
    assert_equal(simpson(y, x=da.linspace(0, 4, 17)).compute(), 32)

    y = da.arange(4)
    x = 2**y
    assert_equal(simpson(y, x=x, even="avg").compute(), 13.875)
    assert_equal(simpson(y, x=x, even="first").compute(), 13.75)
    assert_equal(simpson(y, x=x, even="last").compute(), 14)

    # Tests for checking base case
    x = da.array([3])
    y = da.power(x, 2)
    assert_equal(simpson(y, x=x, axis=0).compute(), 0.0)
    assert_equal(simpson(y, x=x, axis=-1).compute(), 0.0)

    x = da.array([3, 3, 3, 3])
    y = da.power(x, 2)
    assert_equal(simpson(y, x=x, axis=0).compute(), 0.0)
    assert_equal(simpson(y, x=x, axis=-1).compute(), 0.0)

    x = da.array([[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]])
    y = da.power(x, 2)
    zero_axis = [0.0, 0.0, 0.0, 0.0]
    default_axis = [175.75, 175.75, 175.75]
    assert_equal(simpson(y, x=x, axis=0).compute(), zero_axis)
    assert_equal(simpson(y, x=x, axis=-1).compute(), default_axis)

    x = da.array([[1, 2, 4, 8], [1, 2, 4, 8], [1, 8, 16, 32]])
    y = da.power(x, 2)
    zero_axis = [0.0, 136.0, 1088.0, 8704.0]
    default_axis = [175.75, 175.75, 11292.25]
    assert_equal(simpson(y, x=x, axis=0).compute(), zero_axis)
    assert_equal(simpson(y, x=x, axis=-1).compute(), default_axis)

    with pytest.raises(ValueError, match="shape of x must be"):
        simpson(y=da.random.random((5, 5, 5)), x=da.random.random((5, 5))).compute()

    with pytest.raises(ValueError, match="shape of x must be"):
        simpson(y=da.random.random((5,)), x=da.random.random((5, 5))).compute()

    with pytest.raises(ValueError, match="length of x along axis must be"):
        simpson(y=da.arange(10), x=da.arange(15)).compute()

    with pytest.raises(ValueError, match="Parameter 'even' must be"):
        simpson(y=da.arange(10), x=da.arange(10), even="whatever").compute()


def test_romb():
    assert_equal(romb(da.arange(17)).compute(), 128)


def test_newton_cotes():
    """_summary_"""
    # Test the first few degrees, for evenly spaced points
    n = 1
    wts, errcoff = newton_cotes(n, 1)
    assert_equal(wts.compute(), n * da.array([0.5, 0.5]))
    assert_almost_equal(errcoff, -(n**3) / 12.0)

    n = 2
    wts, errcoff = newton_cotes(n, 1)
    assert_almost_equal(wts.compute(), n * da.array([1.0, 4.0, 1.0]) / 6.0)
    assert_almost_equal(errcoff, -(n**5) / 2880.0)

    n = 3
    wts, errcoff = newton_cotes(n, 1)
    assert_almost_equal(wts.compute(), n * da.array([1.0, 3.0, 3.0, 1.0]) / 8.0)
    assert_almost_equal(errcoff, -(n**5) / 6480.0)

    n = 4
    wts, errcoff = newton_cotes(n, 1)
    assert_almost_equal(wts.compute(), n * da.array([7.0, 32.0, 12.0, 32.0, 7.0]) / 90.0)
    assert_almost_equal(errcoff, -(n**7) / 1935360.0)

    # TODO: add test witn n = da.arange(4), equal = 1 to cover `try - if equal`
    # TODO: add test for `try - elif da.all(da.diff(rn) == 1)`
    # TODO: add tests with n = 16, 17 to cover `(N % 2 == 0) and equal` and opp

    # Test newton_cotes with points that are not evenly spaced
    x = da.array([0.0, 1.5, 2.0])
    y = x**2
    wts, errcoff = newton_cotes(x)
    exact_integral = 8.0 / 3
    numeric_integral = da.dot(wts.compute(), y)
    assert_almost_equal(numeric_integral, exact_integral)

    x = da.array([0.0, 1.4, 2.1, 3.0])
    y = x**2
    wts, errcoff = newton_cotes(x)
    exact_integral = 9.0
    numeric_integral = da.dot(wts.compute(), y)
    assert_almost_equal(numeric_integral, exact_integral)

    x = da.array([1.4, 2.1, 3.0])
    with pytest.raises(ValueError, match="start at 0 and end at N"):
        wts, errcoff = newton_cotes(x)

    x = da.array([0.0, 1.4, 2.1, 2.5])
    with pytest.raises(ValueError, match="start at 0 and end at N"):
        wts, errcoff = newton_cotes(x)
