import fftw3
import numpy as np

def _next_pow_two(n):
    n = int(n)

    m = 1
    while m < n:
        m = m << 1

    return m

def transit_decrements(ts, P, T0, Tdur):
    """Returns an array of either ``0`` (out of transit) or ``-1`` (in
    transit) for each time in ``ts``.

    """

    ts_offset = ts - T0
    ts_mod = np.fmod(ts_offset, P)
    ts_mod[ts_mod < 0] += P

    decs = np.zeros(ts_mod.shape[0])
    decs[ts_mod < Tdur] = -1.0

    return decs

def single_transit_decrements(ts, T0, Tdur):
    dts = ts - T0

    tr = np.zeros(ts.shape[0])

    tr[(dts > 0) & (dts < Tdur)] = -1.0

    return tr

def timeshifted_inner_products(x, y):
    r"""Returns an array, :math:`a_i`, where

    .. math::

      a_i = \sum{j} x_j y_{j-i}

    where the indexing :math:`y_{j-i}` represents a cyclic shift of
    :math:`y`. 

    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    n = x.shape[0]
    n2 = 2*n

    m = _next_pow_two(n2)

    xe = np.zeros(m)
    xe[:n] = x

    ye = np.zeros(m)
    ye[:n] = y
    
    xt = np.fft.fft(xe)
    yt = np.fft.fft(ye)

    result = np.real(np.fft.fft(np.conj(xt)*yt))/m

    return result[:n]

def single_transit_timeshifted_inner_products(x, ts, Tdur):
    """Computes the inner product of the array ``x`` with timeshifted
    single transit signals duration ``Tdur``.  The output array gives
    the inner product with the transit decrement that starts at the
    corresponding sample.

    This function should be called instead of generating a decrement
    manually and feeding it to :func:`timeshifted_inner_products`
    because it properly handles issues of wraparound in the
    timeshifting.
    """
    tr = single_transit_decrements(ts, ts[0], Tdur)

    return timeshifted_inner_products(x, tr)

def single_transit_depth_sigma_timeshifts(lc, elc, Tdur):
    ts = lc[:,0]

    t = single_transit_decrements(lc[:,0], lc[0,0], Tdur)

    # Mt.shape == (Nt, Nb)
    Mt = []
    for e in elc:
        Mt.append(single_transit_timeshifted_inner_products(e, ts, Tdur))
    Mt = np.array(Mt)
    Mt = Mt.T

    dd = np.dot(lc[:,1], lc[:,1])

    # dt.shape == (Nt,)
    dt = single_transit_timeshifted_inner_products(lc[:,1], ts, Tdur)

    # tt.shape == (Nt,)
    tt = np.dot(t, t)

    # Md.shape == (Nb,)
    Md = np.dot(elc, lc[:,1])

    # U.shape == (Nt, Nb)
    U = Mt/np.sqrt(tt)

    rhs = Md[np.newaxis, :] - dt[:,np.newaxis]*Mt/tt

    # Sherman-Morrison-Woodbury; we want to solve (1 - U*U^T)*bmax = rhs
    term = np.sum(U*rhs, axis=1)
    term = term / (1 - np.sum(U*U, axis=1))
    bmax = rhs + U*term[:,np.newaxis]

    amax = (dt - np.sum(Mt*bmax, axis=1))/tt

    logl = dd - 2*amax*dt - 2*np.sum(bmax*Md[np.newaxis,:], axis=1) + amax*amax*tt + 2*amax*np.sum(Mt*bmax, axis=1) + np.sum(bmax*bmax, axis=1)

    detC = tt - np.sum(Mt*Mt, axis=1)

    sigma_amax = np.sqrt(1.0/(tt - np.sum(Mt*Mt, axis=1)))

    sigma = np.sqrt(logl / (lc.shape[0]))

    return amax, sigma*sigma_amax, bmax
