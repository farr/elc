#!/usr/bin/env python

import argparse
import astropy.io.fits as fits
import elc
import glob
import h5py
import numpy as np
import os
import os.path as path
import subprocess
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

# Durations from 1 to 32 hours, spaced by sqrt(2) factors
default_durs = np.sqrt(2.0)**np.arange(0, 11)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a directory of lightcurves into HDF5 single-transit depth file')

    parser.add_argument('--outdir', default='depths', help='output directory name (default: %(default)s)')
    parser.add_argument('--lcdir', default='lightcurves', help='lightcurve directory (default: %(default)s)')
    parser.add_argument('--dur', default=default_durs, type=float, action='append', help='duration in hours (default: %(default)s)')

    args = parser.parse_args()

    cwd = os.getcwd()
    try:
        os.chdir('/Users/farr/Documents/Research/KeplerTrend/code')
        git_hash = subprocess.check_output(['git', 'show-ref', '--head', '--hash', 'HEAD'])
    finally:
        os.chdir(cwd)

    files = get_files(args.lcdir)

    outdir = args.outdir

    if os.path.exists(outdir):
        pass
    else:
        os.makedirs(outdir)

    tdurs = np.array(args.dur) / 24.0 # tdur in days

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
        outname = basename.replace('llc', 'std.hdf5')

        if os.path.exists(os.path.join(outdir, outname)):
            pass
        else:
            lc = elc.interpolate_lightcurve(elc.extract_lightcurve(f))

            hfile = h5py.File(os.path.join(outdir, 'temp.hdf5'), 'w')
            try:
                hfile.create_dataset('time', compression='gzip', data=lc[:,0])
                hfile.create_dataset('lightcurve', compression='gzip', data=lc[:,1])
                hfile.create_dataset('tdur', compression='gzip', data=tdurs)

                dgroup = hfile.create_group('depths')
                dugroup = hfile.create_group('depth_uncerts')

                for td in tdurs:
                    name = '{0:.2f}'.format(24.0*td)
                    amax, sigma_amax, bmax = transit.single_transit_depth_sigma_timeshifts(lc, elcs, td)

                    dgroup.create_dataset(name, compression='gzip', data=amax)
                    dugroup.create_dataset(name, compression='gzip', data=sigma_amax)

                hfile.attrs['kepid'] = get_kepid(f)
                hfile.attrs['tdur_units'] = 'd'
                hfile.attrs['tdur_format'] = '.02f'
                hfile.attrs['time_units'] = 'BJD - BJD_ref'
                hfile.attrs['BJD_ref'] = get_bjdref(f)
                hfile.attrs['git_hash'] = git_hash
            finally:
                hfile.close()

            os.rename(os.path.join(outdir, 'temp.hdf5'),
                      os.path.join(outdir, outname))

            print 'Saved as ', outname
            print
