"""Quadrature integrations"""

import math

import dask.array as da
import numpy as np
from scipy.special import gammaln

__all__ = ["simpson"]


def tupleset(t, i, value):
    lst = list(t)
    lst[i] = value
    return tuple(lst)


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
    slice_step = 2  # take every alternate index like 0, 2, 4 or 1, 3, 5
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, slice_step))
    slice1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, slice_step))
    slice2 = tupleset(slice_all, axis, slice(start + 2, stop + 2, slice_step))

    # Regularly spaced Simpson's rule
    # See `Composite Simpson's rule` in the wiki page
    if x is None:
        result = da.sum(y[slice0] + 4 * y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0

    # Irregularly spaced Simpson's rule
    # See `Composite Simpson's rule for irregularly spaced data`
    # in the wiki page
    else:
        # Account for possibly different spacings
        h = da.diff(x, axis=axis)

        sl0 = tupleset(slice_all, axis, slice(start, stop, slice_step))
        sl1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, slice_step))

        h0 = da.array(h[sl0], dtype="float64")
        h1 = da.array(h[sl1], dtype="float64")

        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = da.true_divide(h0, h1, out=da.zeros_like(h0), where=h1 != 0)

        tmp = (
            hsum
            / 6.0
            * (
                y[slice0]
                * (
                    2.0
                    - da.true_divide(
                        1.0,
                        h0divh1,
                        out=da.zeros_like(h0divh1),
                        where=h0divh1 != 0,
                    )
                )
                + y[slice1]
                * (hsum * da.true_divide(hsum, hprod, out=da.zeros_like(hsum), where=hprod != 0))
                + y[slice2] * (2.0 - h0divh1)
            )
        )
        result = da.sum(tmp, axis=axis)

    return result


def simpson(y, x=None, dx=1.0, axis=-1, even="avg"):
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
    nd = len(y.shape)  # Number of dimensions
    N = y.shape[axis]  # Sample size along axis

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
            raise ValueError("If given, shape of x must be 1-D or the same as y.")

        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the same as y.")

    # Sample size is even i.e. number of intervals is an odd number
    # Simpson's rule doesn't support that, so extra steps.
    if N % 2 == 0:
        val = 0.0
        result = 0.0

        slice1 = (slice(None),) * nd
        slice2 = (slice(None),) * nd

        if even not in ["avg", "last", "first"]:
            raise ValueError("Parameter 'even' must be 'avg', 'last', or 'first'.")

        # Compute using Simpson's rule on first intervals
        if even in ["avg", "first"]:
            slice1 = tupleset(slice1, axis, -1)
            slice2 = tupleset(slice2, axis, -2)

            # Apply Trapezoidal rule on the last interval
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])

            # Apply Simpson's rule on the first N-2 intervals
            result = _basic_simpson(y, 0, N - 3, x, dx, axis)

        # Compute using Simpson's rule on last set of intervals
        if even in ["avg", "last"]:
            slice1 = tupleset(slice1, axis, 0)
            slice2 = tupleset(slice2, axis, 1)

            # Apply Trapezoidal rule on the first interval
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val += 0.5 * first_dx * (y[slice2] + y[slice1])

            # Apply Simpson's rule on the last N-2 intervals
            result += _basic_simpson(y, 1, N - 2, x, dx, axis)

        # Divide by 2 to get the average value: (first + last)/2
        if even == "avg":
            val /= 2.0
            result /= 2.0

        result = result + val

    # Sample size is an odd number i.e. even number of intervals
    # Apply Simpson's rule directly
    else:
        result = _basic_simpson(y, 0, N - 2, x, dx, axis)

    if returnshape:
        x = x.reshape(saveshape)

    return result


# Coefficients for Newton-Cotes quadrature
#
# These are the points being used
#  to construct the local interpolating polynomial
#  a are the weights for Newton-Cotes integration
#  B is the error coefficient.
#  error in these coefficients grows as N gets larger.
#  or as samples are closer and closer together

# You can use maxima to find these rational coefficients
#  for equally spaced data using the commands
#  a(i,N) := (integrate(product(r-j,j,0,i-1)
#    * product(r-j,j,i+1,N),r,0,N) / ((N-i)! * i!) * (-1)^(N-i);
#  Be(N) := N^(N+2)/(N+2)! * (N/(N+3) - sum((i/N)^(N+2)*a(i,N),i,0,N));
#  Bo(N) := N^(N+1)/(N+1)! * (N/(N+2) - sum((i/N)^(N+1)*a(i,N),i,0,N));
#  B(N) := (if (mod(N,2)=0) then Be(N) else Bo(N));
#
# pre-computed for equally-spaced weights
#
# num_a, den_a, int_a, num_B, den_B = _builtincoeffs[N]
#
#  a = num_a*array(int_a)/den_a
#  B = num_B*1.0 / den_B
#
#  integrate(f(x),x,x_0,x_N) = dx*sum(a*f(x_i)) + B*(dx)^(2k+3) f^(2k+2)(x*)
#    where k = N // 2
#
_builtincoeffs = {
    1: (1, 2, [1, 1], -1, 12),
    2: (1, 3, [1, 4, 1], -1, 90),
    3: (3, 8, [1, 3, 3, 1], -3, 80),
    4: (2, 45, [7, 32, 12, 32, 7], -8, 945),
    5: (5, 288, [19, 75, 50, 50, 75, 19], -275, 12096),
    6: (1, 140, [41, 216, 27, 272, 27, 216, 41], -9, 1400),
    7: (
        7,
        17280,
        [751, 3577, 1323, 2989, 2989, 1323, 3577, 751],
        -8183,
        518400,
    ),
    8: (
        4,
        14175,
        [989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989],
        -2368,
        467775,
    ),
    9: (
        9,
        89600,
        [2857, 15741, 1080, 19344, 5778, 5778, 19344, 1080, 15741, 2857],
        -4671,
        394240,
    ),
    10: (
        5,
        299376,
        [
            16067,
            106300,
            -48525,
            272400,
            -260550,
            427368,
            -260550,
            272400,
            -48525,
            106300,
            16067,
        ],
        -673175,
        163459296,
    ),
    11: (
        11,
        87091200,
        [
            2171465,
            13486539,
            -3237113,
            25226685,
            -9595542,
            15493566,
            15493566,
            -9595542,
            25226685,
            -3237113,
            13486539,
            2171465,
        ],
        -2224234463,
        237758976000,
    ),
    12: (
        1,
        5255250,
        [
            1364651,
            9903168,
            -7587864,
            35725120,
            -51491295,
            87516288,
            -87797136,
            87516288,
            -51491295,
            35725120,
            -7587864,
            9903168,
            1364651,
        ],
        -3012,
        875875,
    ),
    13: (
        13,
        402361344000,
        [
            8181904909,
            56280729661,
            -31268252574,
            156074417954,
            -151659573325,
            206683437987,
            -43111992612,
            -43111992612,
            206683437987,
            -151659573325,
            156074417954,
            -31268252574,
            56280729661,
            8181904909,
        ],
        -2639651053,
        344881152000,
    ),
    14: (
        7,
        2501928000,
        [
            90241897,
            710986864,
            -770720657,
            3501442784,
            -6625093363,
            12630121616,
            -16802270373,
            19534438464,
            -16802270373,
            12630121616,
            -6625093363,
            3501442784,
            -770720657,
            710986864,
            90241897,
        ],
        -3740727473,
        1275983280000,
    ),
}


def newton_cotes(rn, equal=0):
    r"""
    Return weights and error coefficient for Newton-Cotes integration.

    Suppose we have (N+1) samples of f at the positions
    x_0, x_1, ..., x_N. Then an N-point Newton-Cotes formula for the
    integral between x_0 and x_N is:

    :math:`\int_{x_0}^{x_N} f(x)dx = \Delta x \sum_{i=0}^{N} a_i f(x_i)
    + B_N (\Delta x)^{N+2} f^{N+1} (\xi)`

    where :math:`\xi \in [x_0,x_N]`
    and :math:`\Delta x = \frac{x_N-x_0}{N}` is the average samples spacing.

    If the samples are equally-spaced and N is even, then the error
    term is :math:`B_N (\Delta x)^{N+3} f^{N+2}(\xi)`.

    Parameters
    ----------
    rn : int
        The integer order for equally-spaced data or the relative positions of
        the samples with the first sample at 0 and the last at N, where N+1 is
        the length of `rn`. N is the order of the Newton-Cotes integration.
    equal : int, optional
        Set to 1 to enforce equally spaced data.

    Returns
    -------
    an : ndarray
        1-D array of weights to apply to the function at the provided sample
        positions.
    B : float
        Error coefficient.

    Examples
    --------
    Compute the integral of sin(x) in [0, :math:`\pi`]:

    >>> from scipy.integrate import newton_cotes
    >>> def f(x):
    ...     return da.sin(x)
    >>> a = 0
    >>> b = da.pi
    >>> exact = 2
    >>> for N in [2, 4, 6, 8, 10]:
    ...     x = da.linspace(a, b, N + 1)
    ...     an, B = newton_cotes(N, 1)
    ...     dx = (b - a) / N
    ...     quad = dx * da.sum(an * f(x))
    ...     error = abs(quad - exact)
    ...     print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))
    ...
     2   2.094395102   9.43951e-02
     4   1.998570732   1.42927e-03
     6   2.000017814   1.78136e-05
     8   1.999999835   1.64725e-07
    10   2.000000001   1.14677e-09

    Notes
    -----
    Normally, the Newton-Cotes rules are used on smaller integration
    regions and a composite rule is used to return the total integral.

    """
    try:
        N = len(rn) - 1
        if equal:
            rn = da.arange(N + 1)
        elif da.all(da.diff(rn) == 1):
            equal = 1
    except Exception:
        N = rn
        rn = da.arange(N + 1)
        equal = 1

    if equal and N in _builtincoeffs:
        na, daa, vi, nb, db = _builtincoeffs[N]
        an = na * da.array(vi, dtype=float) / daa
        return an, float(nb) / db

    if (rn[0] != 0) or (rn[-1] != N):
        raise ValueError("The sample positions must start at 0 and end at N")
    yi = rn / float(N)
    ti = 2 * yi - 1
    nvec = da.arange(N + 1)
    C = ti ** nvec[:, np.newaxis]
    Cinv = da.linalg.inv(C)
    # improve precision of result
    for i in range(2):
        Cinv = 2 * Cinv - Cinv.dot(C).dot(Cinv)
    vec = 2.0 / (nvec[::2] + 1)
    ai = Cinv[:, ::2].dot(vec) * (N / 2.0)

    if (N % 2 == 0) and equal:
        BN = N / (N + 3.0)
        power = N + 2
    else:
        BN = N / (N + 2.0)
        power = N + 1

    BN = BN - da.dot(yi**power, ai)
    p1 = power + 1
    fac = power * math.log(N) - gammaln(p1)
    fac = math.exp(fac)
    return ai, float((BN * fac).compute())
