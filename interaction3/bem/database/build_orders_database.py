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
from itertools import repeat
import os
from contextlib import closing
from tqdm import tqdm

from .. core . fma_tests import calculate_error_measures, nine_point_test as test

# register adapters for sqlite to convert numpy types
sql.register_adapter(np.float64, float)
sql.register_adapter(np.float32, float)
sql.register_adapter(np.int64, int)
sql.register_adapter(np.int32, int)


## PROCESS FUNCTIONS ##

def pass_condition(amp_error, phase_error, amp_bias, phase_bias, tol):
    '''
    Returns if test passes based on specified tolerance.
    '''
    if amp_error <= tol and phase_error <= tol:
        return True

    return False


def breakdown_condition(amp_errors, phase_errors):
    '''
    Returns if test translator has broken down (error will shoot up sharply)
    '''
    dy = np.diff(amp_errors) # first derivative
    dyy = np.diff(dy) # second derivative

    if len(dy) > 5:
        if np.all(dy[-5:] > 10): # if change in first derivative is greater than 10 for the last 5 orders
            if np.all(dyy[-4:] > 0): # if second derivative is positive (concave up) for the last 4 orders
                return True

    return False


def process(proc_args):
    '''
    Worker process. Runs FMA test.
    '''
    f, k, l, dims, rho, c, file, tol = proc_args

    xdim, ydim = dims

    start_order = 1
    search_range = 500

    stop_order = start_order + search_range
    orders = range(start_order, stop_order + 1, 2)

    # keep error history
    amp_errors = list()
    phase_errors = list()

    for i, order in enumerate(orders):

        # run test
        test_results = test(k, xdim, ydim, l, order, order, order, rho, c)
        amp_err, phase_err, amp_bias, phase_bias = calculate_error_measures(*test_results)

        # update error history
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

            raw_order = start_order + int(np.argmin(amp_errors)) * 2
            passed = False
            breakdown = True
            break

        # check terminate condition based on stop order
        if order == stop_order:

            raw_order = start_order + int(np.argmin(amp_errors)) * 2
            passed = False
            breakdown = False

    # write raw order and other data to database
    with closing(sql.connect(file)) as conn:
        update_orders_table(conn, f, k, l, raw_order, breakdown, passed)



## POSTPROCESS FUNCTIONS ##

def enforce_monotone_over_frequency(raw_orders, breakdown):
    '''
    Enforce monotonically increasing condition as a function of frequency, i.e. order should remain the same or
    increase as frequency increases.
    '''
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
    '''
    Remove spikes from raw orders as a function as frequency.
    '''
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


def enforce_monotone_over_level(orders, breakdown):
    '''
    Enforce monotonically increasing condition as a function of level, i.e. order should remain the same or
    increase as level increases.
    '''
    for l in reversed(range(len(orders) - 1)):

        orders1 = orders[l]
        breakdown1 = breakdown[l]
        orders2 = orders[l + 1]

        for i in range(len(orders1)):
            if not breakdown1[i]:
                if orders1[i] < orders2[i]:
                    orders1[i] = orders2[i]


def postprocess(conn, fs, levels):
    '''
    Postprocess raw orders by despiking and enforcing monotone rules.
    '''
    minlevel, maxlevel = levels

    orders_list = list()
    breakdown_list = list()

    for l in range(minlevel, maxlevel + 1):

        query = '''
                SELECT * FROM orders 
                WHERE level=?
                ORDER BY frequency
                '''
        table = pd.read_sql(query, conn, params=[l,])

        raw_orders = table.raw_order
        breakdown = table.breakdown

        orders = enforce_monotone_over_frequency(despike_over_frequency(raw_orders, breakdown), breakdown)

        orders_list.append(orders)
        breakdown_list.append(breakdown)

    if minlevel != maxlevel:
        enforce_monotone_over_level(orders_list, breakdown_list)

    levels_list = list(range(minlevel, maxlevel + 1))
    for i in range(len(orders_list)):

        query = '''
                UPDATE orders
                SET translation_order=?
                WHERE 
                frequency=? AND
                level=?
                '''
        conn.executemany(query, zip(orders_list[i], fs, repeat(levels_list[i])))

    conn.commit()


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
            wavenumber float
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
            INSERT INTO frequencies (frequency, wavenumber)
            VALUES (?, ?)
            '''
    conn.executemany(query, zip(fs, ks))

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


def create_orders_table(conn):
    '''
    '''
    # create table
    # frequency, wavenumber, and level are foreign keys referring to their respective tables; this simplifies query
    # syntax when selecting from table
    query = '''
            CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            frequency float,
            wavenumber float,
            level int,
            translation_order int,
            raw_order int,
            breakdown boolean,
            passed boolean,
            FOREIGN KEY (frequency) REFERENCES frequencies (frequency),
            FOREIGN KEY (wavenumber) REFERENCES frequencies (wavenumber),
            FOREIGN KEY (level) REFERENCES levels (level) 
            )
            '''
    conn.execute(query)

    query = '''
            CREATE INDEX order_index ON orders (frequency, level)
            '''
    conn.execute(query)

    conn.commit()


def update_orders_table(conn, f, k, l, raw_order, breakdown, passed):
    '''
    '''
    # insert new record into table
    query = '''
            INSERT INTO orders (frequency, wavenumber, level, raw_order, breakdown, passed)
            VALUES (?, ?, ?, ?, ?, ?)
            '''
    conn.execute(query, (f, k, l, raw_order, breakdown, passed))

    conn.commit()


## ENTRY POINT ##

def main(**kwargs):
    '''
    '''
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
        kwargs['threads'] = threads

    # path to this module's directory
    module_dir = os.path.dirname(os.path.realpath(__file__))

    # set default file name for database
    if file is None:

        file = os.path.join(module_dir, 'orders_dims_{:0.4f}_{:0.4f}.db'.format(*dims))
        kwargs['file'] = file

    # determine frequencies and wavenumbers
    f_start, f_stop, f_step = freqs
    fs_coarse = np.arange(f_cross, f_stop + f_step, f_step)
    fs_fine = np.arange(f_start, f_cross, f_step / f_multi)

    fs = np.concatenate((fs_fine, fs_coarse), axis=0)
    ks = 2 * np.pi * fs / c

    minlevel, maxlevel = levels
    ls = range(minlevel, maxlevel + 1)

    # Check for existing file
    if os.path.isfile(file):

        response = input('Database ' + str(file) + ' already exists. \nOverwrite (y/n)?')
        if response.lower() in ['y', 'yes']:
            os.remove(file)
        else:
            raise Exception('Database already exists')

    # Make directories if they do not exist
    file_dir = os.path.dirname(file)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # create database
    with closing(sql.connect(file)) as conn:

        create_metadata_table(conn, **kwargs)
        create_frequencies_table(conn, fs, ks)
        create_levels_table(conn, levels)
        create_orders_table(conn)

    try:

        # Start multiprocessing pool and run process
        pool = multiprocessing.Pool(max(threads, maxlevel - 1))
        proc_args = [(f, k, l, dims, rho, c, file, tol) for f, k in zip(fs, ks) for l in ls]
        result = pool.imap_unordered(process, proc_args)

        for r in tqdm(result, desc='Building', total=len(proc_args)):
            pass

        with closing(sql.connect(file)) as conn:
            postprocess(conn, fs, levels)


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
    parser.add_argument('--fcrossover', nargs=1, type=float, default=fcrossover)
    parser.add_argument('-fmultiplier', nargs=1, type=int, default=fmultiplier)
    parser.add_argument('-l', '--levels', nargs=2, type=int, default=levels)
    parser.add_argument('-d', '--dims', nargs=2, default=dims)
    parser.add_argument('--tolerance', nargs=1, type=float, default=tolerance)
    parser.add_argument('--sound-speed', nargs=1, type=float, default=sound_speed)
    parser.add_argument('--density', nargs=1, type=float, default=density)

    args = vars(parser.parse_args())
    main(**args)