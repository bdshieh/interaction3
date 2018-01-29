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
    l, k, dims = proc_args

    xdim, ydim = dims

    breakdown = False
    passed = False
    
    stop_order = start_order + search_range
    orders = range(start_order, stop_order + 1, 2)
    
    amp_errors = np.zeros_like(orders, dtype=np.float64)
    phase_errors = np.zeros_like(orders, dtype=np.float64)
    
    for j, order in enumerate(orders):
        
        amp_error, phase_error, amp_bias, phase_bias = calculate_error_measures(*test(k, xdim, ydim, l, order, order,
                                                                                      order, density, sound_speed))
        
        amp_errors[j] = amp_error
        phase_errors[j] = phase_error
        
        # check pass condition
        if pass_condition(amp_error, phase_error, amp_bias, phase_bias):
            
            raw_order = order
            passed = True
            break
            
        # check breakdown condition
        if breakdown_condition(amp_errors[:j + 1], phase_errors[:j + 1]):
            
            raw_order = start_order + np.argmin(amp_errors[:j + 1]) * 2
            breakdown = True
            break
            
        if order == stop_order:
            raw_order = start_order + np.argmin(amp_errors[:j + 1]) * 2

    despike()
    groom_over_wavenumber()

    append_orders_table()



def groom_over_wavenumber(raw_order, breakdown):
    '''
    '''
    order = np.zeros_like(raw_order)
    
    order[0] = raw_order[0]
    
    for i in range(1, len(raw_order)):
        
        if breakdown[i]:
            
            order[i] = raw_order[i]
            continue
            
        if raw_order[i] < order[i -1]:
            order[i] = order[i - 1]
        else:
            order[i] = raw_order[i]
    
    return order


def despike(raw_order, breakdown):
    '''
    '''
    order = raw_order.copy()
    
    dy = np.diff(raw_order)
    
    for i in range(1, len(dy) - 1):
        
        if dy[i] == 0:
            continue
        
        if not breakdown[i]:
            if np.sign(dy[i]) == -np.sign(dy[i - 1]):
                if np.abs(dy[i]) >= 2:
                    order[i] = order[i + 1]
    
    return order


def create_orders_table(conn, fs, ks):

    table = dict()
    table['frequency'] = fs
    table['wavenumber'] = ks

    pd.DataFrame(table).to_sql('ORDERS', conn, if_exists='replace', index=False)


def update_orders_table(conn, fs, ks, l, orders):

    table = dict()
    table['frequency'] = fs
    table['wavenumber'] = ks
    table['level_' + str(l)] = orders


def create_supplemental_table(conn, fs, ks, l, raw_orders, groomed_orders, breakdown, passed):

    table = dict()
    table['frequency'] = fs
    table['wavenumber'] = ks
    table['raw_order'] = raw_orders
    table['order '] = groomed_orders
    table['breakdown'] = breakdown
    table['passed'] = passed

    table_name = 'LEVEL_' + str(l)
    pd.DataFrame(table).to_sql(table_name, conn, if_exists='replace', index=False)


def groom_over_level():
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


def write_to_file(l, results):
    '''
    '''
    with h5py.File(filepath, 'r+') as root:
        
        key = str(l) + '/' + 'raw_order'
        root.create_dataset(key, data=results['raw_order'], compression='gzip',
            compression_opts=9)
            
        key = str(l) + '/' + 'order'
        root.create_dataset(key, data=results['order'], compression='gzip',
            compression_opts=9)
            
        key = str(l) + '/' + 'breakdown'
        root.create_dataset(key, data=results['breakdown'], compression='gzip',
            compression_opts=9)
        
        key = str(l) + '/' + 'passed'
        root.create_dataset(key, data=results['passed'], compression='gzip',
            compression_opts=9)





def make_filepath(folder, xdim, ydim, error_target):
    '''
    '''
    str_format = 'translation_order_xdim_{:0.4f}m_ydim_{:0.4f}m_tol_{:0.1f}.h5'
    path = os.path.join(folder, str_format.format(xdim, ydim, error_target))
    
    return path

# Read in configuration parameters
start_order = 1
search_range = 500

# import test module and function
tests_module = import_module(test_module)
test = getattr(tests_module, test_function)


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

        # write metadata to database
        create_metadata_table(conn, dims, levels, fs, rho, c)

        # write progress to database
        create_progress_table(conn, fs, ks)

        # Set to iterate over all wavenumbers
        new_ks = ks


    try:

        # Start multiprocessing pool and run process
        pool = multiprocessing.Pool(max(threads, maxlevel - 1))
        proc_args = zip()
        result = pool.imap_unordered(process, proc_args)

        # for l, i, ro, psd, bd in tqdm(result, desc='Progress', total=len(levels) * len(ks)):
        #
        #     if np.all(data[l]['done']):
        #
        #         arg1 = despike(data[l]['raw_order'], data[l]['breakdown'])
        #         arg2 = data[l]['breakdown']
        #         data[l]['order'] = groom_over_wavenumber(arg1, arg2)
        #
        #         write_to_file(l, data[l])

        for r in tqdm(result, desc='Building', total=maxlevel - minlevel + 1):
            pass

        if minlevel != maxlevel:
            groom_over_level()

    except Exception as e:
        print(e)

    finally:

        pool.terminate()
        pool.close()


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