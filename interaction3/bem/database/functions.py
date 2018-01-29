## bem / database / functions.py

import numpy as np
import pandas as pd
import sqlite3 as sql
from scipy.interpolate import interp1d
from contextlib import closing

def get_orders_from_db(file, l):

    with closing(sql.connect(file)) as conn:

        query = '''
                SELECT frequency, translation_order FROM orders
                WHERE level=?
                ORDER BY frequency
                '''
        table = pd.read_sql(query, conn, [l,])

    fs = table['frequency']
    orders = table['translation_order']

    return fs, orders


def get_order(file, f, l):

    interp_func = interp1d(*get_orders_from_db(file, l))
    order = int(interp_func(f))
    if order % 2 == 0:
        order += 1

    return order


def get_translation_from_db(file, f, l, coord):

    x, y, z = coord

    with closing(sql.connect(file)) as conn:

        query = '''
                SELECT ntheta, nphi, translation_real, translation_imag FROM translations 
                WHERE frequency=? 
                AND level=? 
                AND x=? 
                AND y=? 
                AND z=?
                ORDER BY theta, phi
                '''
        table = pd.read_sql(query, conn, params=[f, l, x, y, z])

    ntheta = table['ntheta'][0]
    nphi = table['nphi'][0]

    return np.array(table['translation_real'] + 1j * table['translation_imag']).reshape((ntheta, nphi), order='F')


