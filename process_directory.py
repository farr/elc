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
    files = glob.glob(path.join(dir, '*.fits'))

    return files

def get_kepid(f):
    with fits.open(f) as file:
        id = file[0].header['KEPLERID']

    return id

def get_elcs(dir, Neigen = 100):
    files = get_files(dir)
    pfiles = np.random.permutation(files)

    efiles = pfiles[:10*Neigen]

    ids = [get_kepid(f) for f in efiles]

    return elc.elcs(efiles, Neigen=Neigen), ids

def write_elcs(hfile, elcs, ids):
    hfile.create_dataset('elcs', compression='gzip', data=elcs)
    hfile.create_dataset('elc_input_ids', compression='gzip', data=np.array(ids))
    hfile.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a directory of lightcurves into HDF5 single-transit depth file')

    parser.add_argument('--outfile', default='depths.hdf5', help='output file name (default: %(default)s)')
    parser.add_argument('--lcdir', default='lightcurves', help='lightcurve directory (default: %(default)s)')
    parser.add_argument('--dur', default=12, type=int, help='duration in hours (default: %(default)s)')

    args = parser.parse_args()

    files = get_files(args.lcdir)
    hfile = h5py.File(args.outfile)

    try:
        elcs = hfile['elcs']
        print 'Loaded ELCs from HDF5'
    except:
        elcs, ids = get_elcs(args.lcdir)
        write_elcs(hfile, elcs, ids)
        print 'Computed and saved ELCs'

    group_name = 'depths{:d}'.format(args.dur)

    try:
        group = hfile[group_name]
    except:
        hfile.create_group(group_name)
        group = hfile[group_name]
        group.attrs['tdur'] = args.dur
        group.attrs['tdur_units'] = 'hours'

    for f in files:
        print 'Processing ', f
        
        id = get_kepid(f)

        if str(id) in group:
            pass
        else:
            lc = elc.interpolate_lightcurve(elc.extract_lightcurve(f))
            logl, amax, sigma_amax = transit.loglmax_single_transit_timeshifts(lc, elcs, args.dur)

            if 'temp' in group:
                del group['temp']
            group.create_group('temp')
            lc_group = group['temp']

            lc_group.attrs['kepid'] = id

            lc_group.create_dataset('time', compression='gzip', data=lc[:,0])
            lc_group.create_dataset('sap_flux', compression='gzip', data=lc[:,1])
            lc_group.create_dataset('sap_flux_uncert', compression='gzip', data=lc[:,2])
            lc_group.create_dataset('single_tr_depth', compression='gzip', data=amax)
            lc_group.create_dataset('single_tr_uncert', compression='gzip', data=sigma_amax)

            # Write all this out to disk as "temp"
            hfile.flush()

            # Rename (hopefully sort-of atomic)
            group.move('temp', str(id))

            # And flush again.
            hfile.flush()

            print 'Saved'
            print
