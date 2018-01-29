## bem / database / build_orders_database.py
'''
This script will use the specified tester function to empirically determine the
translation operator order and quadrature order needed to achieve the error
target.

Author: Bernard Shieh (bshieh@gatech.edu)
'''

import numpy as np
import pandas as pd
import sqlite3 as sql
import multiprocessing
from itertools import repeat, product
from importlib import import_module
import os
from tqdm import tqdm

from interaction.tests import calculate_error_measures

## PROCESS FUNCTIONS ##

def pass_condition(amp_error, phase_error, amp_bias, phase_bias, tol):
    '''
    '''
    if amp_error <= tol and phase_error <= tol:
        return True

    return False


def breakdown_condition(amp_errors, phase_errors):
    '''
    '''
    dy = np.diff(amp_errors)
    dyy = np.diff(dy)

    if len(dy) > 5:
        if np.all(dy[-5:] > 10):
            if np.all(dyy[-4:] > 0):
                return True

    return False


def process(proc_args):
    '''
    '''
    f, k, l, dims, rho, c, file, tol = proc_args

    xdim, ydim = dims

    stop_order = start_order + search_range
    orders = range(start_order, stop_order + 1, 2)

    amp_errors = list()
    phase_errors = list()

    for i, order in enumerate(orders):

        test_results = test(k, xdim, ydim, l, order, order, order, rho, c)
        amp_err, phase_err, amp_bias, phase_bias = calculate_error_measures(*test_results)

        amp_errors.append(amp_err)
        phase_errors.append(phase_err)

        # check pass condition
        if pass_condition(amp_err, phase_err, amp_bias, phase_bias, tol):

            raw_order = order
            passed = True
            breakdown = False
            break

        # check breakdown condition
        if breakdown_condition(amp_errors, phase_errors):

            raw_order = start_order + np.argmin(amp_errors) * 2
            passed = False
            breakdown = True
            break

        if order == stop_order:

            raw_order = start_order + np.argmin(amp_errors) * 2
            passed = False
            breakdown = False

    conn = sql.connect(file)
    update_orders_table(conn, f, l, raw_order, breakdown, passed)


## POSTPROCESS FUNCTIONS ##

def enforce_monotone_over_frequency(raw_orders, breakdown):

    orders = np.zeros_like(raw_orders)

    orders[0] = raw_orders[0]

    for i in range(1, len(raw_orders)):

        if breakdown[i]:
            orders[i] = raw_orders[i]
            continue

        if raw_orders[i] < orders[i - 1]:
            orders[i] = orders[i - 1]
        else:
            orders[i] = raw_orders[i]

    return orders


def despike_over_frequency(raw_orders, breakdown):

    orders = raw_orders.copy()

    dy = np.diff(raw_orders)

    for i in range(1, len(dy) - 1):

        if dy[i] == 0:
            continue

        if not breakdown[i]:
            if np.sign(dy[i]) == -np.sign(dy[i - 1]):
                if np.abs(dy[i]) >= 2:
                    orders[i] = orders[i + 1]

    return orders


def enforce_monotone_over_level():
    '''
    '''
    with h5py.File(filepath, 'r+') as root:

        for l in range(maxlevel - 1, minlevel - 1, -1):

            order1 = root[str(l) + '/' + 'order'][:]
            breakdown1 = root[str(l) + '/' + 'breakdown'][:]
            order2 = root[str(l + 1) + '/' + 'order'][:]

            for i in range(len(order1)):
                if not breakdown1[i]:
                    if order1[i] < order2[i]:
                        order1[i] = order2[i]

            root[str(l) + '/' + 'order'][:] = order1


def postprocess(file, levels):

    minlevel, maxlevel = levels
    conn = sql.connect(file)

    for l in range(minlevel, maxlevel + 1):

        query = '''
                SELECT * FROM orders 
                LEFT JOIN frequencies WHERE orders.frequency_id=frequencies.id
                LEFT JOIN levels WHERE orders.level_id=levels.id
                WHERE level=?
                ORDER BY frequency
                '''
        table = pd.read_sql(query, conn, params=[l,])

        raw_orders = table.raw_order
        breakdown = table.breakdown



    pass


## DATABASE FUNCTIONS ##

def create_frequencies_table(conn, fs, ks):

    query = '''
            CREATE TABLE frequencies ( 
            id int PRIMARY KEY,
            frequency float,
            wavenumber float
            )
            '''
    conn.execute(query)

    query = '''
            INSERT INTO frequencies (frequency, wavenumber)
            VALUES (?, ?)
            '''
    conn.executemany(query, zip(fs, ks))


def create_levels_table(conn, levels):

    minlevel, maxlevel = levels

    query = '''
            CREATE TABLE levels (
            id int PRIMARY KEY,
            level int
            )
            '''
    conn.execute(query)

    query = '''
            INSERT INTO levels (level)
            VALUES (?)
            '''
    conn.executemany(query, range(minlevel, maxlevel + 1))


def create_orders_table(conn):

    query = '''
            CREATE TABLE orders (
            id int PRIMARY KEY,
            frequency_id int,
            level_id int,
            order int,
            raw_order int,
            breakdown bool,
            passed bool,
            FOREIGN KEY (frequency_id) REFERENCES frequencies (id),
            FOREIGN KEY (level_id) REFERENCES levels (id) 
            )
            '''
    conn.execute(query)


def update_orders_table(conn, f, l, raw_order, breakdown, passed):

    query = '''
            INSERT INTO orders (frequency_id, level_id, raw_order, breakdown, passed)
            VALUES (SELECT id from frequencies WHERE frequency=?, SELECT id from levels WHERE level=?, ?, ?, ?)
            '''
    conn.execute(query, f, l, raw_order, breakdown, passed)
    # conn.executemany(query, zip(fs, repeat(l), orders, raw_orders, breakdown, passed))


# Read in configuration parameters
start_order = 1
search_range = 500

# import test module and function
tests_module = import_module(test_module)
test = getattr(tests_module, test_function)

## ENTRY POINT ##

def main(**kwargs):

    threads = kwargs['threads']
    freqs = kwargs['freqs']
    f_cross = kwargs['fcrossover']
    f_multi = kwargs['fmultiplier']
    levels = kwargs['levels']
    dims = kwargs['dims']
    tol = kwargs['tolerance']
    c = kwargs['sound_speed']
    rho = kwargs['density']
    file = kwargs['file']

    # set default threads to logical core count
    if threads is None:
        threads = multiprocessing.cpu_count()

    # path to this module's directory
    module_dir = os.path.dirname(os.path.realpath(__file__))

    # set default file name for database
    if file is None:
        file = os.path.join(module_dir, 'orders_dims_{:0.4f}_{:0.4f}.db'.format(*dims))

    # determine frequencies and wavenumbers
    f_start, f_stop, f_step = freqs
    fs_coarse = np.arange(f_cross, f_stop + f_step, f_step)
    fs_fine = np.arange(f_start, f_cross, f_step * f_multi)

    fs = np.concatenate((fs_fine, fs_coarse), axis=0)
    ks = 2 * np.pi * fs / c

    minlevel, maxlevel = levels
    ls = range(minlevel, maxlevel + 1)

    # Check for existing file and existing wavenumbers
    if os.path.isfile(file):

        conn = sql.connect(file)
        existing_ks = pd.read_sql('SELECT wavenumber FROM progress WHERE is_complete=True', conn)

        # Set to iterate over only new wavenumbers
        new_ks = np.array([k for k in ks if k not in existing_ks])
        # new_k = np.array([x for x in k if round(x, 4) not in existing_k])

        # raise Exception('File already exists')

    else:

        # Make directories if they do not exist
        file_dir = os.path.dirname(file)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # create database
        conn = sql.connect(file)

        create_frequencies_table(conn, fs, ks)
        create_levels_table(conn, levels)
        create_orders_table(conn)

        # Set to iterate over all wavenumbers
        new_ks = ks

    try:

        # Start multiprocessing pool and run process
        pool = multiprocessing.Pool(max(threads, maxlevel - 1))
        proc_args = [(f, k, l, dims, rho, c, file, tol) for f, k in zip(fs, ks) for l in ls]
        result = pool.imap_unordered(process, proc_args)

        for r in tqdm(result, desc='Building', total=(maxlevel - minlevel + 1)):
            pass

        postprocess()

        if minlevel != maxlevel:
            groom_over_level()

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
    freqs = 50e3, 50e6, 500e3
    fcrossover = 1e6
    fmultiplier = 10
    levels = 2, 6
    dims = 4e-3, 4e-3
    tolerance = 0.01
    sound_speed = 1500
    density = 1000
    file = None

    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', default=file)
    parser.add_argument('-t', '--threads', nargs=1, type=int, default=nthreads)
    parser.add_argument('-f', '--freqs', nargs=3, type=float, default=freqs)
    parser.add_argument('-fc', '--fcrossover', nargs=1, type=float, default=fcrossover)
    parser.add_argument('-fm', '-fmultiplier', nargs=1, type=int, default=fmultiplier)
    parser.add_argument('-l', '--levels', nargs=2, type=int, default=levels)
    parser.add_argument('-d', '--dims', nargs=2, default=dims)
    parser.add_argument('-t', '--tolerance', nargs=1, type=float, default=tolerance)
    parser.add_argument('--sound-speed', nargs=1, type=float, default=sound_speed)
    parser.add_argument('--density', nargs=1, type=float, default=density)

    args = vars(parser.parse_args())
    main(**args)