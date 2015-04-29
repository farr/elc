#!/usr/bin/env python

import argparse
import astropy.io.fits as fits
import elc
import glob
import h5py
import numpy as np
import os
import os.path as path
import transit

def get_files(dir):
    files = glob.glob(path.join(dir, '*llc.fits'))

    return files

def get_kepid(f):
    with fits.open(f) as file:
        id = file[0].header['KEPLERID']

    return id

def get_bjdref(f):
    with fits.open(f) as file:
        bjdi = file[1].header['BJDREFI']
        bjdf = file[1].header['BJDREFF']

    return bjdi + bjdf

def get_elcs(dir, Neigen = 100):
    files = get_files(dir)
    pfiles = np.random.permutation(files)

    efiles = pfiles[:10*Neigen]

    ids = [get_kepid(f) for f in efiles]

    return elc.elcs(efiles, Neigen=Neigen), ids

def write_elcs(outdir, elcs, ids):
    hfile = h5py.File(os.path.join(outdir, 'elcs.hdf5'), 'w')
    hfile.create_dataset('elcs', compression='gzip', data=elcs)
    hfile.create_dataset('elc_input_ids', compression='gzip', data=np.array(ids))
    hfile.close()

def read_elcs(outdir):
    hfile = h5py.File(os.path.join(outdir, 'elcs.hdf5'), 'r')
    try:
        elcs_dset = hfile['elcs']
        elcs = np.zeros(elcs_dset.shape)
        elcs_dset.read_direct(elcs)
    finally:
        hfile.close()

    return elcs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a directory of lightcurves into HDF5 single-transit depth file')

    parser.add_argument('--outdir', default='depths', help='output directory name (default: %(default)s)')
    parser.add_argument('--lcdir', default='lightcurves', help='lightcurve directory (default: %(default)s)')
    parser.add_argument('--dur', default=12, type=int, help='duration in hours (default: %(default)s)')

    args = parser.parse_args()

    files = get_files(args.lcdir)

    outdir = os.path.join(args.outdir, '{:02d}'.format(args.dur))
    try:
        os.makedirs(outdir)
    except:
        pass

    tdur = args.dur / 24.0 # tdur in days

    try:
        elcs = read_elcs(outdir)
        print 'Loaded ELCs from HDF5'
    except:
        elcs, ids = get_elcs(args.lcdir)
        write_elcs(outdir, elcs, ids)
        print 'Computed and saved ELCs'

    for f in files:
        print 'Processing ', f

        indir, fname = os.path.split(f)
        basename, ext = os.path.splitext(fname)
        outname = basename.replace('llc', 'std{:02d}'.format(args.dur)) + '.hdf5'

        if os.path.exists(os.path.join(outdir, outname)):
            pass
        else:
            lc = elc.interpolate_lightcurve(elc.extract_lightcurve(f))
            logl, amax, sigma_amax = transit.loglmax_single_transit_timeshifts(lc, elcs, tdur)

            hfile = h5py.File(os.path.join(outdir, 'temp.hdf5'), 'w')
            try:
                hfile.create_dataset('time', compression='gzip', data=lc[:,0])
                hfile.create_dataset('depth', compression='gzip', data=amax)
                hfile.create_dataset('depth_uncert', compression='gzip', data=sigma_amax)
                hfile.attrs['kepid'] = get_kepid(f)
                hfile.attrs['tdur'] = tdur
                hfile.attrs['tdur_units'] = 'd'
                hfile.attrs['time_units'] = 'BJD - BJD_ref'
                hfile.attrs['BJD_ref'] = get_bjdref(f)
            finally:
                hfile.close()

            os.rename(os.path.join(outdir, 'temp.hdf5'),
                      os.path.join(outdir, outname))

            print 'Saved as ', outname
            print
