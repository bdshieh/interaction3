## mfield / simulations / sim_functions.py

import numpy as np
import pandas as pd
import scipy.signal
import itertools
from contextlib import closing
from itertools import repeat
import sqlite3 as sql


def meshview(v1, v2, v3, mode='cartesian', as_list=True):

    if mode.lower() in ('cart', 'cartesian', 'rect'):

        x, y, z = np.meshgrid(v1, v2, v3, indexing='ij')

    elif mode.lower() in ('spherical', 'sphere', 'polar'):

        r, theta, phi = np.meshgrid(v1, v2, v3, indexing='ij')

        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)

    elif mode.lower() in ('sector', 'sec'):

        r, py, px = np.meshgrid(v1, v2, v3, indexing='ij')

        px = -px
        pyp = np.arctan(np.cos(px) * np.sin(py) / np.cos(py))

        x = r * np.sin(pyp)
        y = -r * np.cos(pyp) * np.sin(px)
        z = r * np.cos(px) * np.cos(pyp)

    if as_list:
        return np.c_[x.ravel(order='F'), y.ravel(order='F'), z.ravel(order='F')]
    else:
        return x, y, z


def cart2sec(xyz):

    x, y, z = np.atleast_2d(xyz).T

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    pyp = np.arcsin(x / r)
    px = np.arcsin(-y / r / np.cos(pyp))
    py = np.arctan(np.tan(pyp) / np.cos(px))

    return np.c_[r, py, -px]


# def concatenate_with_padding(rf_data, t0s, fs):
#
#     mint0 = min(t0s)
#
#     frontpads = [int(np.ceil((t - mint0) * fs)) for t in t0s]
#     maxlen = max([fpad + len(rf) for fpad, rf in zip(frontpads, rf_data)])
#     backpads = [maxlen - (fpad + len(rf)) for fpad, rf in zip(frontpads, rf_data)]
#
#     new_data = []
#
#     for rf, fpad, bpad in zip(rf_data, frontpads, backpads):
#
#         new_rf = np.pad(rf, ((0,0), (fpad, bpad)), mode='constant')
#         new_data.append(new_rf)
#
#     return np.array(new_data), mint0


def concatenate_with_padding(rf_data, t0s, fs, axis=-1):

    if len(rf_data) <= 1:
        return np.atleast_2d(rf_data), t0s[0]

    rf_data = np.atleast_2d(*rf_data)

    mint0 = float(min(t0s))
    frontpads = [int(np.ceil((t - mint0) * fs)) for t in t0s]
    maxlen = max([fpad + rf.shape[1] for fpad, rf in zip(frontpads, rf_data)])
    backpads = [maxlen - (fpad + rf.shape[1]) for fpad, rf in zip(frontpads, rf_data)]

    new_data = []

    for rf, fpad, bpad in zip(rf_data, frontpads, backpads):

        new_rf = np.pad(rf, ((0,0), (fpad, bpad)), mode='constant')
        new_data.append(new_rf)

    if axis == 2:
        return np.stack(new_data, axis=axis), mint0
    else:
        return np.concatenate(new_data, axis=axis), mint0


def sum_with_padding(rf_data, t0s, fs):

    if len(rf_data) <= 1:
        return np.atleast_2d(rf_data[0]), t0s[0]

    rf_data = np.atleast_2d(*rf_data)

    mint0 = min(t0s)
    frontpads = [int(np.ceil((t - mint0) * fs)) for t in t0s]
    maxlen = max([fpad + rf.shape[1] for fpad, rf in zip(frontpads, rf_data)])
    backpads = [maxlen - (fpad + rf.shape[1]) for fpad, rf in zip(frontpads, rf_data)]

    new_data = []

    for rf, fpad, bpad in zip(rf_data, frontpads, backpads):

        new_rf = np.pad(rf, ((0,0), (fpad, bpad)), mode='constant')
        new_data.append(new_rf)

    return np.sum(new_data, axis=0), mint0


def gausspulse(fc, fbw, fs):

    cutoff = scipy.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=-100, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs

    t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
    pulse, _ = scipy.signal.gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)

    return pulse, t


def envelope(rf_data, axis=-1):
    return np.abs(scipy.signal.hilbert(np.atleast_2d(rf_data), axis=axis))


def chunks(iterable, n):

    res = []
    for el in iterable:
        res.append(el)
        if len(res) == n:
            yield res
            res = []
    if res:
        yield res


def create_jobs(*args, mode='zip', is_complete=None):
    '''
    Convenience function for creating jobs (sets of input arguments) for multiprocessing Pool. Supports zip and product
    combinations, and automatic chunking of iterables.
    '''
    static_args = list()
    static_idx = list()
    iterable_args = list()
    iterable_idx = list()

    for arg_no, arg in enumerate(args):
        if isinstance(arg, (tuple, list)):

            iterable, chunksize = arg
            iterable_args.append(chunks(iterable, chunksize))
            iterable_idx.append(arg_no)
        else:

            static_args.append(itertools.repeat(arg))
            static_idx.append(arg_no)

    if not iterable_args and not static_args:
        return

    if not iterable_args:
        yield 1, tuple(args[i] for i in static_idx)

    if not static_args:
        repeats = itertools.repeat(())
    else:
        repeats = zip(*static_args)

    if mode.lower() == 'product':
        combos = itertools.product(*iterable_args)
    elif mode.lower() == 'zip':
        combos = zip(*iterable_args)
    elif mode.lower() == 'zip_longest':
        combos = itertools.zip_longest(*iterable_args)

    for job_id, (r, p) in enumerate(zip(repeats, combos)):

        # skip jobs that have been completed
        if is_complete is not None and is_complete[job_id]:
            continue

        res = r + p
        # reorder vals according to input order
        yield job_id, tuple(res[i] for i in np.argsort(static_idx + iterable_idx))


def rotation_matrix(vec, angle):

    if isinstance(vec, str):
        string = vec.lower()
        if string == 'x':
            vec = [1, 0, 0]
        elif string == '-x':
            vec = [-1, 0, 0]
        elif string == 'y':
            vec = [0, 1, 0]
        elif string == '-y':
            vec = [0, -1, 0]
        elif string == 'z':
            vec = [0, 0, 1]
        elif string == '-x':
            vec = [0, 0, -1]

    x, y, z = vec
    a = angle

    r = np.zeros((3, 3))
    r[0, 0] = np.cos(a) + x**2 * (1 - np.cos(a))
    r[0, 1] = x * y * (1 - np.cos(a)) - z * np.sin(a)
    r[0, 2] = x * z * (1 - np.cos(a)) + y * np.sin(a)
    r[1, 0] = y * x * (1 - np.cos(a)) + z * np.sin(a)
    r[1, 1] = np.cos(a) + y**2 * (1 - np.cos(a))
    r[1, 2] = y * z * (1 - np.cos(a)) - x * np.sin(a)
    r[2, 0] = z * x * (1 - np.cos(a)) - z * np.sin(a)
    r[2, 1] = z * y * (1 - np.cos(a)) + x * np.sin(a)
    r[2, 2] = np.cos(a) + z**2 * (1 - np.cos(a))

    return r


def rotate_nodes(nodes, vec, angle):

    rmatrix = rotation_matrix(vec, angle)
    return rmatrix.dot(nodes.T).T


## DATABASE FUNCTIONS ##

def open_sqlite_file(f):

    def decorator(firstarg, *args, **kwargs):

        if isinstance(firstarg, sql.Connection):
            return f(firstarg, *args, **kwargs)
        else:
            with closing(sql.connect(firstarg)) as con:
                return f(con, *args, **kwargs)

    return decorator


@open_sqlite_file
def table_exists(con, name):

    query = '''SELECT count(*) FROM sqlite_master WHERE type='table' and name=?'''
    return con.execute(query, (name,)).fetchone()[0] != 0


@open_sqlite_file
def create_metadata_table(con, **kwargs):

    table = [[str(v) for v in list(kwargs.values())]]
    columns = list(kwargs.keys())
    pd.DataFrame(table, columns=columns, dtype=str).to_sql('metadata', con, if_exists='replace', index=False)


@open_sqlite_file
def create_progress_table(con, njobs):

    with con:
        # create table
        con.execute('CREATE TABLE progress (job_id INTEGER PRIMARY KEY, is_complete boolean)')
        # insert values
        con.executemany('INSERT INTO progress (is_complete) VALUES (?)', repeat((False,), njobs))


@open_sqlite_file
def get_progress(con):

    table = pd.read_sql('SELECT is_complete FROM progress SORT BY job_id', con)

    is_complete = np.array(table).squeeze()
    ijob = sum(is_complete)

    return is_complete, ijob


@open_sqlite_file
def update_progress(con, job_id):

    with con:
        con.execute('UPDATE progress SET is_complete=1 WHERE job_id=?', [job_id,])