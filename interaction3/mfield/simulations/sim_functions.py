## mfield / simulations / sim_functions.py

import numpy as np
import pandas as pd
import scipy.signal


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


def concatenate_with_padding(rf_data, t0s, fs):

    mint0 = min(t0s)

    frontpads = [int(np.ceil((t - mint0) * fs)) for t in t0s]
    maxlen = max([fpad + len(rf) for fpad, rf in zip(frontpads, rf_data)])
    backpads = [maxlen - (fpad + len(rf)) for fpad, rf in zip(frontpads, rf_data)]

    new_data = []

    for rf, fpad, bpad in zip(rf_data, frontpads, backpads):

        new_rf = np.pad(rf, (fpad, bpad), mode='constant')
        new_data.append(new_rf)

    return np.array(new_data), mint0


def gausspulse(fc, fbw, fs):

    cutoff = scipy.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=-100, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs

    t = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
    pulse, _ = scipy.signal.gausspulse(t, fc=fc, bw=fbw, retquad=True, bwr=-3)

    return pulse, t


def chunks(iterable, n):

    res = []
    for el in iterable:
        res.append(el)
        if len(res) == n:
            yield res
            res = []
    if res:
        yield res


def rotation_matrix(vec, angle):

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


## DATABSE FUNCTIONS ##

def table_exists(con, name):

    query = '''SELECT count(*) FROM sqlite_master WHERE type='table' and name=?'''
    return con.execute(query, (name,)).fetchone()[0] != 0


def create_metadata_table(con, **kwargs):

    table = [[str(v) for v in list(kwargs.values())]]
    columns = list(kwargs.keys())

    pd.DataFrame(table, columns=columns, dtype=str).to_sql('metadata', con, if_exists='replace', index=False)
