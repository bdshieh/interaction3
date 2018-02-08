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
from contextlib import closing
import os
from tqdm import tqdm

from interaction3.bem.core import fma_functions as fma
from interaction3.bem.core.db_functions import get_order

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)


## PROCESS FUNCTIONS ##

# def get_orders_interp_funcs(orders_db, levels):
#
#     minlevel, maxlevel = levels
#     orders_interp_funcs = dict()
#     conn = sql.connect(orders_db)
#
#     for l in range(minlevel, maxlevel + 1):
#
#         query = '''
#                 SELECT wavenumber, translation_order FROM orders
#                 WHERE level=?
#                 ORDER BY wavenumber
#                 '''
#         table = pd.read_sql(query, conn, params=[l,])
#
#         ks = table['wavenumber']
#         orders = table['translation_order']
#
#         orders_interp_funcs[l] = interp1d(ks, orders)
#
#     return orders_interp_funcs


def generate_translations(file, f, k, dims, levels, orders_db):

    xdim, ydim = dims
    minlevel, maxlevel = levels

    for l in range(minlevel, maxlevel + 1):

        order = get_order(orders_db, f, l)

        qrule = fma.fft_quadrule(order, order)
        group_xdim, group_ydim = xdim / (2 ** l), ydim / (2 ** l)

        kcoordT = qrule['kcoordT']
        theta = qrule['theta']
        phi = qrule['phi']

        unique_coords = fma.get_unique_coords()

        for coords in unique_coords:

            r = coords * np.array([group_xdim, group_ydim, 1])
            rhat = r / fma.mag(r)
            cos_angle = rhat.dot(kcoordT)

            translation = np.ascontiguousarray(fma.mod_ff2nf_op(fma.mag(r), cos_angle, k, order))
            with closing(sql.connect(file)) as conn:
                update_translations_table(conn, f, k, l, order, tuple(coords), theta, phi, translation)


def process(proc_args):

    file, f, k, dims, levels, orders_db = proc_args

    generate_translations(file, f, k, dims, levels, orders_db)

    with closing(sql.connect(file)) as conn:
        update_progress(conn, f)


## DATABASE FUNCTIONS ##

def create_metadata_table(conn, **kwargs):
    '''
    '''
    table = [[str(v) for v in list(kwargs.values())]]
    columns = list(kwargs.keys())

    pd.DataFrame(table, columns=columns, dtype=str).to_sql('metadata', conn, if_exists='replace', index=False)


def create_frequencies_table(conn, fs, ks):
    '''
    '''
    # create table
    query = '''
            CREATE TABLE frequencies ( 
            frequency float,
            wavenumber float,
            is_complete boolean
            )
            '''
    conn.execute(query)

    # create unique index on frequency
    query = '''
            CREATE UNIQUE INDEX frequency_index ON frequencies (frequency)
            '''
    conn.execute(query)

    # create unique index on wavenumber
    query = '''
            CREATE UNIQUE INDEX wavenumber_index ON frequencies (wavenumber)
            '''
    conn.execute(query)

    # insert values into table
    query = '''
            INSERT INTO frequencies (frequency, wavenumber, is_complete)
            VALUES (?, ?, ?)
            '''
    conn.executemany(query, zip(fs, ks, repeat(False)))

    conn.commit()


def update_progress(conn, f):

    query = '''
            UPDATE frequencies SET is_complete=1 WHERE frequency=?
            '''
    conn.execute(query, [f,])

    conn.commit()


def create_levels_table(conn, levels):
    '''
    '''
    minlevel, maxlevel = levels

    # create table
    query = '''
            CREATE TABLE levels (
            level int
            )
            '''
    conn.execute(query)

    # create unique index on level
    query = '''
            CREATE UNIQUE INDEX level_index ON levels (level)
            '''
    conn.execute(query)

    # insert values into table
    query = '''
            INSERT INTO levels (level)
            VALUES (?)
            '''
    conn.executemany(query, list((x,) for x in range(minlevel, maxlevel + 1)))

    conn.commit()


def create_coordinates_table(conn):

    unique_coords = fma.get_unique_coords()

    query = '''
            CREATE TABLE coordinates (
            x int,
            y int,
            z int
            )
            '''
    conn.execute(query)

    query = '''
            CREATE UNIQUE INDEX coordinates_index ON coordinates (x, y, z)
            '''
    conn.execute(query)

    query = '''
            INSERT INTO coordinates
            VALUES (?, ?, ?)
            '''
    conn.executemany(query, unique_coords)

    conn.commit()


def create_translations_table(conn):

    query = '''
            CREATE TABLE translations (
            id INTEGER PRIMARY KEY,
            frequency float,
            wavenumber float,
            level int,
            x int,
            y int,
            z int,
            theta float,
            phi float,
            ntheta int,
            nphi int,
            translation_order int,
            translation_real float,
            translation_imag float,
            FOREIGN KEY (frequency) REFERENCES frequencies (frequency),
            FOREIGN KEY (wavenumber) REFERENCES frequencies (wavenumber),
            FOREIGN KEY (level) REFERENCES levels (level),
            FOREIGN KEY (x, y, z) REFERENCES coordinates (x, y, z)
            )
            '''
    conn.execute(query)

    query = '''
            CREATE INDEX translation_index ON translations (frequency, level, x, y, z)
            '''
    conn.execute(query)

    conn.commit()


def update_translations_table(conn, f, k, l, order, coord, thetas, phis, translations):

    x, y, z = coord
    thetav, phiv = np.meshgrid(thetas, phis, indexing='ij')
    ntheta = len(thetas)
    nphi = len(phis)

    query = '''
            INSERT INTO translations (frequency, wavenumber, level, x, y, z, ntheta, nphi, theta, phi, 
            translation_order, translation_real, translation_imag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
    conn.executemany(query, zip(repeat(f), repeat(k), repeat(l), repeat(x), repeat(y), repeat(z), repeat(ntheta),
                                repeat(nphi), thetav.ravel(), phiv.ravel(), repeat(order),
                                np.real(translations.ravel()), np.imag(translations.ravel())))

    conn.commit()


## ENTRY POINT ##

def main(**kwargs):

    threads = kwargs['threads']
    freqs = kwargs['freqs']
    levels = kwargs['levels']
    dims = kwargs['dims']
    c = kwargs['sound_speed']
    file = kwargs['file']
    orders_db = kwargs['orders_db']

    # set default threads to logical core count
    if threads is None:

        threads = multiprocessing.cpu_count()
        kwargs['threads'] = threads

    # path to this module's directory
    module_dir = os.path.dirname(os.path.realpath(__file__))

    # set default file name for database
    if file is None:

        file = os.path.join(module_dir, 'translations_dims_{:0.4f}_{:0.4f}.db'.format(*dims))
        kwargs['file'] = file

    # set default file nam of orders database to use
    if orders_db is None:

        orders_db = os.path.join(module_dir, 'orders_dims_{:0.4f}_{:0.4f}.db'.format(*dims))
        kwargs['orders_db'] = orders_db

    # read orders database and form interpolating functions
    # orders_interp_funcs = get_orders_interp_funcs(orders_db, levels)

    # check for existing file
    if os.path.isfile(file):

        # conn = sql.connect(file)
        response = input('Database ' + str(file) + ' already exists. \nContinue (c), Overwrite (o), or Do nothing ('
                                                   'any other key)?')

        if response.lower() in ['o', 'overwrite']:

            os.remove(file)

            # determine frequencies and wavenumbers
            f_start, f_stop, f_step = freqs
            fs = np.arange(f_start, f_stop + f_step, f_step)
            ks = 2 * np.pi * fs / c

            # create database
            with closing(sql.connect(file)) as conn:

                # create database tables
                create_metadata_table(conn, **kwargs)
                create_frequencies_table(conn, fs, ks)
                create_levels_table(conn, levels)
                create_coordinates_table(conn)
                create_translations_table(conn)

        elif response.lower() in ['c', 'continue']:

            with closing(sql.connect(file)) as conn:

                query = '''
                        SELECT (frequency, wavenumber) FROM frequencies 
                        WHERE is_complete=False
                        '''
                table = pd.read_sql(query, conn)

            fs = np.array(table['frequency'])
            ks = np.array(table['wavenumber'])

        else:
            raise Exception('Database already exists')

    else:

        # Make directories if they do not exist
        file_dir = os.path.dirname(file)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # determine frequencies and wavenumbers
        f_start, f_stop, f_step = freqs
        fs = np.arange(f_start, f_stop + f_step, f_step)
        ks = 2 * np.pi * fs / c

        # create database
        with closing(sql.connect(file)) as conn:

            # create database tables
            create_metadata_table(conn, **kwargs)
            create_frequencies_table(conn, fs, ks)
            create_levels_table(conn, levels)
            create_coordinates_table(conn)
            create_translations_table(conn)

    try:

        # start multiprocessing pool and run process
        pool = multiprocessing.Pool(threads)
        proc_args = [(file, f, k, dims, levels, orders_db) for f, k in zip(fs, ks)]
        result = pool.imap_unordered(process, proc_args)

        for r in tqdm(result, desc='Building', total=len(fs)):
            pass

    except Exception as e:
        print(e)

    finally:

        pool.terminate()
        pool.close()


## COMMAND LINE INTERFACE ##

if __name__ == '__main__':

    import argparse

    # default arguments
    nthreads = None
    freqs = 50e3, 50e6, 50e3
    levels = 2, 6
    dims = 4e-3, 4e-3
    sound_speed = 1500
    file = None
    orders_db = None

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default=file)
    parser.add_argument('-t', '--threads', type=int, default=nthreads)
    parser.add_argument('-f', '--freqs', nargs=3, type=float, default=freqs)
    parser.add_argument('-l', '--levels', nargs=2, type=int, default=levels)
    parser.add_argument('-d', '--dims', nargs=2, type=float, default=dims)
    parser.add_argument('-o', '--orders-db', type=str, default=orders_db)
    parser.add_argument('--sound-speed', type=float, default=sound_speed)

    args = vars(parser.parse_args())
    main(**args)

            