## bem / simulations / functions.py

import numpy as np
import itertools
import pandas as pd
from contextlib import closing
import sqlite3 as sql
import functools
from itertools import repeat

from .. core import bem_functions as bem


## CONNECTORS ##

def connector_square_cmut_membrane(obj, **kwargs):

    length_x = obj['length_x']
    length_y = obj['length_y']
    electrode_x = obj['electrode_x']
    electrode_y = obj['electrode_y']
    nnodes_x = obj['nnodes_x']
    nnodes_y = obj['nnodes_y']
    position = obj['position']
    rotations = obj.get('rotations', None)
    mem_rho = obj['density']

    f = kwargs['frequency']
    rho = kwargs['density']
    c = kwargs['sound_speed']
    use_preconditioner = kwargs['use_preconditioner']
    dc_bias = kwargs['dc_bias']

    nodes_at_origin, nnodes, dx, dy, s_n, e_mask = generate_square_nodes(length_x, length_y, electrode_x, electrode_y,
                                                                         nnodes_x, nnodes_y)

    # apply rotations to nodes if specified
    if rotations is not None:
        for vec, angle in rotations:
            nodes_at_origin = rotate_nodes(nodes_at_origin, vec, angle)

    # translate to membrane center
    nodes = nodes_at_origin + position

    # create matrices
    # mass matrix
    h = obj['thickness']
    M = bem.m_matrix(mem_rho, h, nnodes)

    # damping matrix
    att_mech = obj['att_mech']
    B = bem.b_matrix(att_mech, nnodes)

    # stiffness matrix
    if 'k_matrix_comsol_file' in obj:

        filename = obj['k_matrix_comsol_file']
        K = bem.k_matrix_comsol(filename)

    else:

        E = obj['y_modulus']
        eta = obj['p_ratio']
        shape = nnodes_x - 2, nnodes_y - 2
        K = bem.k_matrix_fd2(E, h, eta, dx, dy, nnodes, shape)

    # spring-softening matrix
    h_gap = obj['gap']
    h_isol = obj['isolation']
    e_r = obj['permittivity']
    Kss, is_collapsed, u0, t_ratios = bem.kss_matrix(K, e_mask, dc_bias, h_gap, h_isol, e_r, nnodes)
    if is_collapsed:
        raise Exception

    # single membrane acoustic impedance matrix
    if use_preconditioner:
        Zr1 = bem.zr1_matrix(nodes_at_origin, f, rho, c, s_n)

    output = dict()
    output['nodes'] = nodes
    output['nnodes'] = nnodes
    output['dx'] = dx
    output['dy'] = dy
    output['node_area'] = s_n
    output['M'] = M
    output['B'] = B
    output['K'] = K
    output['Kss'] = Kss
    if use_preconditioner:
        output['Zr1'] = Zr1
    output['electrode_mask'] = e_mask
    output['static_displacement'] = u0
    output['transformer_ratios'] = t_ratios

    return output


def connector_circular_cmut_membrane(obj, **kwargs):

    radius = obj['radius']
    electrode_r = obj['electrode_r']
    nnodes_x = obj['nnodes_x']
    nnodes_y = obj['nnodes_y']
    center = obj['center']
    rotations = obj.get('rotations', None)

    f = kwargs['frequency']
    rho = kwargs['density']
    c = kwargs['sound_speed']
    use_preconditioner = kwargs['use_preconditioner']
    dc_bias = kwargs['dc_bias']

    nodes_at_origin, nnodes, dx, dy, s_n, e_mask = generate_circular_nodes(radius, electrode_r, nnodes_x, nnodes_y)

    # apply rotations to nodes if specified
    if rotations is not None:
        for vec, angle in rotations:
            nodes_at_origin = rotate_nodes(nodes_at_origin, vec, angle)

    # translate to membrane center
    nodes = nodes_at_origin + center

    # create matrices
    # mass matrix
    h = obj['thickness']
    M = bem.m_matrix(rho, h, nnodes)

    # damping matrix
    att_mech = obj['att_mech']
    B = bem.b_matrix(att_mech, nnodes)

    # stiffness matrix
    if 'k_matrix_comsol_file' in obj:
        filename = obj['k_matrix_comsol_file']
        K = bem.k_matrix_comsol(filename)
    else:
        raise NotImplementedError # finite-differences not supported for circular membranes yet

    # spring-softening matrix
    h_gap = obj['gap']
    h_isol = obj['isolation']
    e_r = obj['permittivity']
    Kss, is_collapsed, u0, t_ratios = bem.kss_matrix(K, e_mask, dc_bias, h_gap, h_isol, e_r, nnodes)
    if is_collapsed:
        raise Exception

    # single membrane acoustic impedance matrix
    if use_preconditioner:
        Zr1 = bem.zr1_matrix(nodes_at_origin, f, rho, c, s_n)

    output = dict()
    output['nodes'] = nodes
    output['nnodes'] = nnodes
    output['dx'] = dx
    output['dy'] = dy
    output['node_area'] = s_n
    output['M'] = M
    output['B'] = B
    output['K'] = K
    output['Kss'] = Kss
    if use_preconditioner:
        output['Zr1'] = Zr1
    output['electrode_mask'] = e_mask
    output['static_displacement'] = u0
    output['transformer_ratios'] = t_ratios

    return output


def connector_square_pmut_membrane(obj, **kwargs):

    length_x = obj['length_x']
    length_y = obj['length_y']
    electrode_x = obj['electrode_x']
    electrode_y = obj['electrode_y']
    nnodes_x = obj['nnodes_x']
    nnodes_y = obj['nnodes_y']
    center = obj['center']
    rotations = obj.get('rotations', None)

    f = kwargs['frequency']
    rho = kwargs['density']
    c = kwargs['sound_speed']
    use_preconditioner = kwargs['use_preconditioner']

    nodes_at_origin, nnodes, dx, dy, s_n, e_mask = generate_square_nodes(length_x, length_y, electrode_x, electrode_y,
                                                                          nnodes_x, nnodes_y)

    # apply rotations to nodes if specified
    if rotations is not None:
        for vec, angle in rotations:
            nodes_at_origin = rotate_nodes(nodes_at_origin, vec, angle)

    # translate to membrane center
    nodes = nodes_at_origin + center

    # create matrices
    # mass matrix
    h = obj['thickness']
    M = bem.m_matrix(rho, h, nnodes)

    # damping matrix
    att_mech = obj['att_mech']
    B = bem.b_matrix(att_mech, nnodes)

    # stiffness matrix
    if 'k_matrix_comsol_file' in obj:
        filename = obj['k_matrix_comsol_file']
        K = bem.k_matrix_comsol(filename)
    else:
        E = obj['y_modulus']
        eta = obj['p_ratio']
        shape = nnodes_x - 2, nnodes_y - 2
        K = bem.k_matrix_fd2(E, h, eta, dx, dy, nnodes, shape)

    # single membrane acoustic impedance matrix
    if use_preconditioner:
        Zr1 = bem.zr1_matrix(nodes_at_origin, f, rho, c, s_n)

    # piezoelectric actuating load matrix
    filename = obj['peq_matrix_file']
    Peq = bem.peq_matrix(filename)

    output = dict()
    output['nodes'] = nodes
    output['nnodes'] = nnodes
    output['dx'] = dx
    output['dy'] = dy
    output['node_area'] = s_n
    output['M'] = M
    output['B'] = B
    output['K'] = K
    if use_preconditioner:
        output['Zr1'] = Zr1
    output['Peq'] = Peq
    output['electrode_mask'] = e_mask

    return output


def connector_circular_pmut_membrane(obj, **kwargs):

    radius = obj['radius']
    electrode_r = obj['electrode_r']
    nnodes_x = obj['nnodes_x']
    nnodes_y = obj['nnodes_y']
    center = obj['center']
    rotations = obj.get('rotations', None)

    f = kwargs['frequency']
    rho = kwargs['density']
    c = kwargs['sound_speed']
    use_preconditioner = kwargs['use_preconditioner']

    nodes_at_origin, nnodes, dx, dy, s_n, e_mask = generate_circular_nodes(radius, electrode_r, nnodes_x, nnodes_y)

    # apply rotations to nodes if specified
    if rotations is not None:
        for vec, angle in rotations:
            nodes_at_origin = rotate_nodes(nodes_at_origin, vec, angle)

    # translate to membrane center
    nodes = nodes_at_origin + center

    # create matrices
    # mass matrix
    h = obj['thickness']
    M = bem.m_matrix(rho, h, nnodes)

    # damping matrix
    att_mech = obj['att_mech']
    B = bem.b_matrix(att_mech, nnodes)

    # stiffness matrix
    if 'k_matrix_comsol_file' in obj:
        filename = obj['k_matrix_comsol_file']
        K = bem.k_matrix_comsol(filename)
    else:
        E = obj['y_modulus']
        eta = obj['p_ratio']
        shape = nnodes_x - 2, nnodes_y - 2
        K = bem.k_matrix_fd2(E, h, eta, dx, dy, nnodes, shape)

    # single membrane acoustic impedance matrix
    if use_preconditioner:
        Zr1 = bem.zr1_matrix(nodes_at_origin, f, rho, c, s_n)

    # piezoelectric actuating load matrix
    filename = obj['peq_matrix_file']
    Peq = bem.peq_matrix(filename)

    output = dict()
    output['nodes'] = nodes
    output['nnodes'] = nnodes
    output['dx'] = dx
    output['dy'] = dy
    output['node_area'] = s_n
    output['M'] = M
    output['B'] = B
    output['K'] = K
    if use_preconditioner:
        output['Zr1'] = Zr1
    output['Peq'] = Peq
    output['electrode_mask'] = e_mask

    return output


def connector_channel():
    pass


def connector_defocused_channel():
    pass


## HELPER FUNCTIONS ##

def generate_square_nodes(length_x, length_y, electrode_x, electrode_y, nnodes_x, nnodes_y):

    # define nodes on x-y plane
    xv = np.linspace(-length_x / 2, length_x / 2, nnodes_x)
    yv = np.linspace(-length_y / 2, length_y / 2, nnodes_y)
    zv = 0
    x, y, z = np.meshgrid(xv[1:-1], yv[1:-1], zv) # remove boundary nodes
    nodes = np.c_[x.ravel(), y.ravel(), z.ravel()]
    nnodes = len(nodes)

    # calculate node spacing and node area
    dx = length_x / (nnodes_x - 1)
    dy = length_y / (nnodes_y - 1)
    s_n = dx * dy

    # flag whether node is in electrode or not
    x_condition = np.abs(nodes[:, 0]) <= electrode_x / 2.
    y_condition = np.abs(nodes[:, 1]) <= electrode_y / 2.
    e_mask = np.logical_and(x_condition, y_condition)

    return nodes, nnodes, dx, dy, s_n, e_mask


def generate_circular_nodes(radius, electrode_r, nnodes_x, nnodes_y):

    # define nodes on x-y plane
    length_x = radius * 2
    length_y = radius * 2
    xv = np.linspace(-length_x / 2, length_x / 2, nnodes_x)
    yv = np.linspace(-length_y / 2, length_y / 2, nnodes_y)
    zv = 0
    x, y, z = np.meshgrid(xv[1:-1], yv[1:-1], zv)  # remove boundary nodes
    nodes = np.c_[x.ravel(), y.ravel(), z.ravel()]

    # calculate node spacing and node area
    dx = length_x / (nnodes_x - 1)
    dy = length_y / (nnodes_y - 1)
    s_n = dx * dy

    # keep only nodes inside radius with edge buffer
    edge_buffer = np.sqrt((dx / 2) ** 2 + (dy / 2) ** 2)
    r = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
    edge_mask = r <= radius - edge_buffer
    nodes = nodes[edge_mask, :]
    nnodes = len(nodes)

    # flag whether node is in electrode or not
    r = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
    e_mask = r <= electrode_r

    return nodes, nnodes, dx, dy, s_n, e_mask


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


## DATABASE FUNCTIONS ##

def open_sqlite_file(f):

    @functools.wraps
    def decorator(file, *args, **kwargs):

        if isinstance(file, sql.Connection):
            return f(file, *args, **kwargs)
        else:
            with closing(sql.connect(file)) as con:
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