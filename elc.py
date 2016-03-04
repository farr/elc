import astropy.io.fits as af
import numpy as np

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
    
    sel = np.isnan(ys)
    
    ys[sel] = 0.0
    dys[sel] = 1.0
    
    return np.column_stack((all_ts, ys, dys))

def elcs(filelist, Neigen):
    """Using SVD, produces a set of eigen-lightcurves from the given
    list of FITS files.

    """

    lcmatrix = []
    for f in filelist:
        lc = interpolate_lightcurve(extract_lightcurve(f))
        # Normalise to have the same integrated flux
        lc[:,1] /= sum(lc[:,1])
        lcmatrix.append(lc[:,1])
    lcmatrix = np.array(lcmatrix)
    u, s, v = np.linalg.svd(lcmatrix)

    return v[:Neigen,:]
