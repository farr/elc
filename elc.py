import astropy.io.fits as af
import numpy as np
import scipy.interpolate as si

def extract_lightcurve(file):
    """Given a FITS filename containing a Kepler lightcurve, extracts
    a ``(n, 3)`` shaped array containing the times, fluxes
    (``SAP_FLUX``), and flux uncertainties (``SAP_FLUX_ERR``) from the
    file.

    """ 
    header = af.open(file)
    ts = header[1].data['TIME']
    ys = header[1].data['SAP_FLUX']
    dys = header[1].data['SAP_FLUX_ERR']
    
    return np.column_stack((ts, ys, dys))

def interpolate_lightcurve(lc):
    """Produces a linearly-interpolated lightcurve, filling in any
    ``NaN`` gaps.
    
    """
    
    ts = lc[:,0]
    ys = lc[:,1]
    dys = lc[:,2]
    
    all_ts = np.linspace(ts[0], ts[-1], ts.shape[0])
    
    sel = ~np.isnan(ys)
    
    ys = si.interp1d(ts[sel], ys[sel])(all_ts)
    dys = si.interp1d(ts[sel], dys[sel])(all_ts)
    
    return np.column_stack((all_ts, ys, dys))

def elcs(filelist, Neigen):
    """Using SVD, produces a set of eigen-lightcurves from the given
    list of FITS files.

    """

    lcmatrix = []
    for f in filelist:
        lc = interpolate_lightcurve(extract_lightcurve(f))
        # Normalise
        lc[:,1] -= np.mean(lc[:,1])
        lc[:,1] /= np.std(lc[:,1])
        lcmatrix.append(lc[:,1])
    lcmatrix = np.array(lcmatrix)
    u, s, v = np.linalg.svd(lcmatrix)

    # Insert the row of ones at the top of v
    v = np.row_stack((np.ones(v.shape[1]), v[:Neigen,:]))

    uu, ss, vv = np.linalg.svd(v)
    

    return vv[:(Neigen+1), :]
