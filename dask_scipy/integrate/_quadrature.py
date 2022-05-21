"""
"""

import dask.array as da

__all__ = ['simpson']


def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def trapezoid():
    pass


def _basic_simpson(y, start, stop, x, dx, axis):
    """This is the implementation of Simpson's composite rules for 
    regularly and irregularly spaced data. Please refer to the 
    following wiki page for more details:

        https://en.wikipedia.org/wiki/Simpson%27s_rule

    Note: this is not a public function.

    Args:
        y (da.array): f(x)
        start (int/float): Interval lower bound.
        stop (int/float): Interval upper bound.
        x (da.array): x
        dx (int/float): Step size of sub intervals.
        axis (int): Calculation along axis.

    Returns:
        float: Simpson's approximation of the given integration.
    """
    nd = len(y.shape)

    # Calculate Simpson's the 3 slices for composites rules
    slice_step = 2 # take every alternate index like 0, 2, 4 or 1, 3, 5
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, slice_step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, slice_step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, slice_step))

    # Regularly spaced Simpson's rule 
    # See `Composite Simpson's rule` in the wiki page
    if x is None:
        result = da.sum(y[slice0] + 4*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    
    # Irregularly spaced Simpson's rule 
    # See `Composite Simpson's rule for irregularly spaced data` in the wiki page
    else:
        # Account for possibly different spacings
        h = da.diff(x, axis=axis)

        sl0 = tupleset(slice_all, axis, slice(start, stop, slice_step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, slice_step))

        h0 = da.array(h[sl0], dtype='float64')
        h1 = da.array(h[sl1], dtype='float64')

        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = da.true_divide(h0, h1, out=da.zeros_like(h0), where=h1 != 0)

        tmp = hsum/6.0 * (
            y[slice0] * (
                2.0 - da.true_divide(
                    1.0, h0divh1, 
                    out=da.zeros_like(h0divh1), 
                    where=h0divh1 != 0
                )
            ) 
            + y[slice1] * (
                hsum 
                * da.true_divide(
                    hsum, hprod, 
                    out=da.zeros_like(hsum), 
                    where=hprod != 0
                )
            ) 
            + y[slice2] * (2.0 - h0divh1)
        )
        result = da.sum(tmp, axis=axis)

    return result


def simpson(y, x=None, dx=1.0, axis=-1, even='avg'):
    """
    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule. If x is None, spacing of dx is assumed.

    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals. The parameter 'even' controls how this is handled.

    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : float, optional
        Spacing of integration points along axis of `x`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.
    even : str {'avg', 'first', 'last'}, optional
        'avg' : Average two results:
                1) use the first N-2 intervals with
                  a trapezoidal rule on the last interval
                2) use the last N-2 intervals with 
                  a trapezoidal rule on the first interval.

        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.

        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.

    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less. If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.

    Examples
    --------
    >>> from dask-scipy import integrate
    >>> x = da.arange(0, 10)
    >>> y = da.arange(0, 10)

    >>> integrate.simpson(y, x).compute()
    40.5

    >>> y = da.power(x, 3)
    >>> integrate.simpson(y, x).compute()
    1642.5
    >>> integrate.quad(lambda x: x**3, 0, 9)[0]
    1640.25

    >>> integrate.simpson(y, x, even='first').compute()
    1644.5

    """
    y = da.asarray(y)
    nd = len(y.shape) # Number of dimensions
    N = y.shape[axis] # Sample size along axis

    last_dx = dx
    first_dx = dx
    returnshape = 0
    
    if x is not None:
        x = da.asarray(x)

        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))

        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")

        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    # Sample size is even i.e. number of intervals is an odd number
    # Simpson's rule doesn't support that, so extra steps.
    if N % 2 == 0:
        val = 0.0
        result = 0.0

        slice1 = (slice(None),) * nd
        slice2 = (slice(None),) * nd

        if even not in ['avg', 'last', 'first']:
            raise ValueError("Parameter 'even' must be 'avg', 'last', or 'first'.")

        # Compute using Simpson's rule on first intervals
        if even in ['avg', 'first']:
            slice1 = tupleset(slice1, axis, -1)
            slice2 = tupleset(slice2, axis, -2)

            # Apply Trapezoidal rule on the last interval
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5*last_dx*(y[slice1]+y[slice2])

            # Apply Simpson's rule on the first N-2 intervals
            result = _basic_simpson(y, 0, N-3, x, dx, axis)

        # Compute using Simpson's rule on last set of intervals
        if even in ['avg', 'last']:
            slice1 = tupleset(slice1, axis, 0)
            slice2 = tupleset(slice2, axis, 1)

            # Apply Trapezoidal rule on the first interval
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val += 0.5*first_dx*(y[slice2]+y[slice1])

            # Apply Simpson's rule on the last N-2 intervals
            result += _basic_simpson(y, 1, N-2, x, dx, axis)

        # Divide by 2 to get the average value: (first + last)/2
        if even == 'avg':
            val /= 2.0
            result /= 2.0

        result = result + val

    # Sample size is an odd number i.e. even number of intervals
    # Apply Simpson's rule directly
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)

    if returnshape:
        x = x.reshape(saveshape)

    return result

# if __name__ == "__main__":
#     x = da.arange(1000000001)
#     y = da.square(x)

#     print(_basic_simpson(y=y, start=0, stop=1000000000, x=x, dx=1, axis=-1).compute())