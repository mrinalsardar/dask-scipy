import dask.array as da
from numpy.testing import assert_equal

from dask_scipy.integrate import simpson


def test_simpson():
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
