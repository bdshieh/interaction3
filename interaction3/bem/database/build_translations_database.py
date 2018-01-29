## interaction / scripts / create_translation_repository.py
'''
This script will pre-calculate the translation operators for a given bounding
box, max level, and frequency steps for a multi-level fast multipole algorithm.
This can take hours to days depending on the number of threads available, size 
of bounding box, number of levels etc. The output is stored in a single H5 file.

To use, run with a corresponding yaml config file for setting the input 
parameters.

python create_translation_repository.py <path to config file>

Author: Bernard Shieh (bshieh@gatech.edu)
'''

import numpy as np
import pandas as pd
import sqlite3 as sql
import multiprocessing
from itertools import repeat
from scipy.interpolate import interp1d
import os
from tqdm import tqdm
import h5py

from . import fma_functions as fma


def generate_translations(f, k, dims, levels, orders_interp_funcs, conn):

    xdim, ydim = dims
    minlevel, maxlevel = levels

    def mag(r):
        return np.sqrt(np.sum(r ** 2))

    orders = list()

    for l in range(minlevel, maxlevel + 1):

        order = int(orders_interp_funcs[l](k))
        if order % 2 == 0:
            order += 1
        orders.append(order)

        qrule = fma.fft_quadrule(order, order)
        group_xdim, group_ydim = xdim / (2 ** l), ydim / (2 ** l)

        kcoordT = qrule['kcoord'].transpose((0, 2, 1))
        theta = qrule['theta']
        phi = qrule['phi']

        x, y, z = np.mgrid[1:4, 0:4, 0:1:1j]
        uniques = np.c_[x.ravel(), y.ravel(), z.ravel()].astype(int)
        uniques = uniques[2:, :]

        for coords in uniques:

            r = coords * np.array([group_xdim, group_ydim, 1])
            rhat = r / mag(r)
            cos_angle = rhat.dot(kcoordT)

            translation = np.ascontiguousarray(fma.mod_ff2nf_op(mag(r), cos_angle, k, qrule))
            append_translations_table(conn, f, k, l, tuple(coords), theta, phi, translation)

    append_orders_table(conn, f, k, list(range(minlevel, maxlevel + 1)), orders)


def create_metadata_table(conn, dims, levels, freqs, rho, c):

    xdim, ydim = dims
    f_start, f_stop, f_step = freqs
    minlevel, maxlevel = levels

    table = [xdim, ydim, minlevel, maxlevel, f_start, f_stop, f_step, rho, c]
    columns = ['x_dimension', 'y_dimension', 'minimum_level', 'maximum_level', 'frequency_start', 'frequency_stop',
               'frequency_step', 'density', 'sound_speed']

    pd.DataFrame(table, columns=columns).to_sql('METADATA', conn, if_exists='replace', index=False)


def append_orders_table(conn, f, k, levels, orders):

    table = [f, k] + list(orders)
    columns = ['frequency', 'wavenumber'] + ['level_' + str(l) for l in list(levels)]

    pd.DataFrame(table, columns=columns).to_sql('ORDERS', conn, if_exists='append', index=False)


def append_translations_table(conn, f, k, l, coords, theta, phi, translation):

    x, y, z = coords
    thetav, phiv = np.meshgrid(theta, phi, indexing='ij')

    table = dict()
    table['frequency'] = f
    table['wavenumber'] = k
    table['x'] = int(x)
    table['y'] = int(y)
    table['z'] = int(z)
    table['theta'] = thetav.ravel()
    table['phi'] = phiv.ravel()
    table['translation_real'] = np.real(translation.ravel())
    table['translation_imag'] = np.imag(translation.ravel())
    table['level'] = int(l)

    pd.DataFrame(table).to_sql('TRANSLATIONS', conn, if_exists='append', index=False)


def create_progress_table(conn, frequencies, wavenumbers):

    table = dict()
    table['frequency'] = frequencies
    table['wavenumber'] = wavenumbers
    table['is_complete'] = False

    pd.DataFrame(table).to_sql(conn, 'PROGRESS', if_exists='replace', index=False)


def update_progress_table(conn, k):
    conn.execute('UPDATE progress SET is_complete=True WHERE wavenumber=?', k)



def get_orders_interp_funcs(orders_db, levels):

    minlevel, maxlevel = levels

    orders_interp_funcs = dict()

    with h5py.File(orders_db, 'r') as root:

        ks = root['wavenumbers'][:]

        for l in range(minlevel, maxlevel + 1):

            orders = root[str(l) + '/' + 'order'][:]
            orders_interp_funcs[l] = interp1d(ks, orders)

    return orders_interp_funcs


def default_file_path(xdim, ydim):
    
    str_format = 'translations_dims_{:0.4f}_{:0.4f}'
    return str_format.format(xdim, ydim)


def process(proc_args):

    f, k, dims, levels, orders_interp_funcs, conn = proc_args

    generate_translations(f, k, dims, levels, orders_interp_funcs, conn)
    update_progress_table(conn, k)


def main(**kwargs):

    threads = kwargs['threads']
    freqs = kwargs['freqs']
    levels = kwargs['levels']
    dims = kwargs['dims']
    c = kwargs['sound_speed']
    rho = kwargs['density']
    file = kwargs['file']
    orders_db = kwargs['orders_db']

    # set default threads to logical core count
    if threads is None:
        threads = multiprocessing.cpu_count()

    # path to this module's directory
    module_dir = os.path.dirname(os.path.realpath(__file__))

    # set default file name for database
    if file is None:
        file = os.path.join(module_dir, 'translations_dims_{:0.4f}_{:0.4f}.db'.format(*dims))

    # set default file nam of orders database to use
    if orders_db is None:
        orders_db = os.path.join(module_dir, 'orders_dims_{:0.4f}_{:0.4f}.db'.format(*dims))

    # determine frequencies and wavenumbers
    f_start, f_stop, f_step = freqs
    fs = np.arange(f_start, f_stop + f_step, f_step)
    ks = 2 * np.pi * fs / c

    # read orders database and form interpolating functions
    orders_interp_funcs = get_orders_interp_funcs(orders_db, levels)

    # check for existing file
    if os.path.isfile(file):

        conn = sql.connect(file)
        existing_ks = pd.read_sql('SELECT wavenumber FROM progress WHERE is_complete=True', conn)

        # Set to iterate over only new wavenumbers
        new_ks = np.array([k for k in ks if k not in existing_ks])
        # new_k = np.array([x for x in k if round(x, 4) not in existing_k])

    else:

        # Make directories if they do not exist
        file_dir = os.path.dirname(file)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # create database
        conn = sql.connect(file)

        # write metadata to database
        create_metadata_table(conn, dims, levels, fs, rho, c)

        # write progress to database
        create_progress_table(conn, fs, ks)

        # Set to iterate over all wavenumbers
        new_ks = ks

    try:

        # start multiprocessing pool and run process
        pool = multiprocessing.Pool(threads)
        proc_args = zip(fs, ks, repeat(dims), repeat(levels), repeat(orders_interp_funcs), repeat(conn))
        result = pool.imap_unordered(process, proc_args)

        for r in tqdm(result, desc='Building', total=len(new_ks)):
            pass

    except Exception as e:
        print(e)

    finally:

        pool.terminate()
        pool.close()


if __name__ == '__main__':

    import argparse

    # default arguments
    nthreads = None
    freqs = 50e3, 50e6, 50e3
    levels = 2, 6
    dims = 4e-3, 4e-3
    sound_speed = 1500
    density = 1000
    file = None
    orders_db = None

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default=file)
    parser.add_argument('-t', '--threads', nargs=1, type=int, default=nthreads)
    parser.add_argument('-f', '--freqs', nargs=3, type=float, default=freqs)
    parser.add_argument('-l', '--levels', nargs=2, type=int, default=levels)
    parser.add_argument('-d', '--dims', nargs=2, default=dims)
    parser.add_argument('-o', '--orders-db', nargs=1, type=str, default=orders_db)
    parser.add_argument('--sound-speed', nargs=1, type=float, default=sound_speed)
    parser.add_argument('--density', nargs=1, type=float, default=density)

    args = vars(parser.parse_args())
    main(**args)

            