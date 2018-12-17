## interaction3 / arrays / foldable_spiral.py

import numpy as np
from scipy.spatial.distance import cdist as distance
from scipy.optimize import brentq

from interaction3.abstract import *
from interaction3 import util


def blackman_functions(R):

    cos = np.cos
    sin = np.sin
    pi = np.pi

    integral = lambda r: (R ** 2 * (0.5 * cos(pi * r / R) + 0.02 * cos(2.0 * pi * r / R) - 0.52) + pi * R * r * (
                0.5 * sin(pi * r / R) + 0.04 * sin(2.0 * pi * r / R)) + 0.21 * pi ** 2 * r ** 2) / pi ** 2
    a_eff = (-1.0 * R ** 2 + 0.21 * pi ** 2 * R ** 2) / pi ** 2

    return integral, a_eff


def _get_blackman_symbolic_equations():

    import sympy

    cos = sympy.cos
    pi = sympy.pi
    r, R = sympy.symbols('r R')

    A = 0.42 - 0.5 * cos(2 * pi *(r / (2 * R) + 0.5)) + 0.08 * cos(4 * pi * (r / (2 * R) + 0.5))
    integral = sympy.simplify(sympy.integrate(A * r, (r, 0, r)))
    a_eff = integral.subs(r, R)

    return integral, a_eff


def main(cfg, args):

    nelem = cfg.nelem
    nmem_x, nmem_y = cfg.nmem
    mempitch_x, mempitch_y = cfg.mempitch
    length_x, length_y = cfg.length
    electrode_x, electrode_y = cfg.electrode
    nnodes_x, nnodes_y = cfg.nnodes
    ndiv_x, ndiv_y = cfg.ndiv
    edge_buffer = cfg.edge_buffer
    taper_radius = cfg.taper_radius

    # calculated parameters
    gr = np.pi * (np.sqrt(5) - 1)
    integral, a_eff = blackman_functions(taper_radius)

    # membrane properties
    memprops = {}
    memprops['length_x'] = length_x
    memprops['length_y'] = length_y
    memprops['electrode_x'] = electrode_x
    memprops['electrode_y'] = electrode_y
    memprops['y_modulus'] = cfg.y_modulus
    memprops['p_ratio'] = cfg.p_ratio
    memprops['isolation'] = cfg.isolation
    memprops['permittivity'] = cfg.permittivity
    memprops['gap'] = cfg.gap
    memprops['nnodes_x'] = nnodes_x
    memprops['nnodes_y'] = nnodes_y
    memprops['thickness'] = cfg.thickness
    memprops['density'] = cfg.density
    memprops['att_mech'] = cfg.att_mech
    memprops['ndiv_x'] = ndiv_x
    memprops['ndiv_y'] = ndiv_y

    # calculate membrane positions
    xx, yy, zz = np.meshgrid(np.linspace(0, (nmem_x - 1) * mempitch_x, nmem_x),
                             np.linspace(0, (nmem_y - 1) * mempitch_y, nmem_y),
                             0)
    mem_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] - [(nmem_x - 1) * mempitch_x / 2,
                                                           (nmem_y - 1) * mempitch_y / 2,
                                                           0]

    # define transmit element centers
    elem_pos = []
    for n in range(nelem):

        f = lambda r: integral(r) - (2 * n + 1) * a_eff / (2 * nelem)
        r = brentq(f, 0, taper_radius)
        theta = (n + 1) * gr
        xx = r * np.sin(theta)
        yy = r * np.cos(theta)
        zz = 0
        elem_pos.append([xx, yy, zz])

    elem_pos = np.array(elem_pos)

    # create arrays, bounding box and rotation points are hard-coded
    vertices = [[-3.75e-3, -3.75e-3, 0],
                [-3.75e-3, 3.75e-3, 0],
                [-1.25e-3, 3.75e-3, 0],
                [-1.25e-3, -3.75e-3, 0]]
    x0, y0, _ = vertices[0]
    x1, y1, _ = vertices[2]
    xx, yy, zz = elem_pos.T
    mask = np.logical_and(np.logical_and(np.logical_and(xx >= (x0 + edge_buffer), xx < (x1 - edge_buffer)),
                                         yy >= (y0 + edge_buffer)), yy < (y1 - edge_buffer))
    array0 = _construct_array(0, np.array([-1.25e-3, 0, 0]), vertices, elem_pos[mask, :], mem_pos, memprops)

    vertices = [[-1.25e-3, -3.75e-3, 0],
                [-1.25e-3, 3.75e-3, 0],
                [1.25e-3, 3.75e-3, 0],
                [1.25e-3, -3.75e-3, 0]]
    x0, y0, _ = vertices[0]
    x1, y1, _ = vertices[2]
    xx, yy, zz = elem_pos.T
    mask = np.logical_and(np.logical_and(np.logical_and(xx >= (x0 + edge_buffer), xx < (x1 - edge_buffer)),
                                         yy >= (y0 + edge_buffer)), yy < (y1 - edge_buffer))
    array1 = _construct_array(1, np.array([0, 0, 0]), vertices, elem_pos[mask, :], mem_pos, memprops)

    vertices = [[1.25e-3, -3.75e-3, 0],
                [1.25e-3, 3.75e-3, 0],
                [3.75e-3, 3.75e-3, 0],
                [3.75e-3, -3.75e-3, 0]]
    x0, y0, _ = vertices[0]
    x1, y1, _ = vertices[2]
    xx, yy, zz = elem_pos.T
    mask = np.logical_and(np.logical_and(np.logical_and(xx >= (x0 + edge_buffer), xx < (x1 - edge_buffer)),
                                         yy >= (y0 + edge_buffer)), yy < (y1 - edge_buffer))
    array2 = _construct_array(2, np.array([1.25e-3, 0, 0]), vertices, elem_pos[mask, :], mem_pos, memprops)

    return array0, array1, array2


def _construct_array(id, rotation_origin, vertices, elem_pos, mem_pos, memprops):

    if rotation_origin is None:
        rotation_origin = np.array([0,0,0])

    # construct channels
    elements = []
    mem_counter = 0
    elem_counter = 0

    for epos in elem_pos:
        membranes = []
        for mpos in mem_pos:
            # construct membrane
            m = SquareCmutMembrane(**memprops)
            m.id = mem_counter
            m.position = (epos + mpos).tolist()

            membranes.append(m)
            mem_counter += 1

        # construct element
        elem = Element()
        elem.id = elem_counter
        elem.kind = 'both'
        elem.dc_bias = 0
        elem.active = True
        elem.delay = 0
        elem.position = epos.tolist()
        elem.membranes = membranes
        element_position_from_membranes(elem)

        elements.append(elem)
        elem_counter += 1

    # construct array
    array = Array()
    array.id = id
    array.elements = elements,
    array.rotation_origin = rotation_origin.tolist()
    array.vertices = vertices
    array_position_from_vertices(array)

    return array


# default parameters
_Config = {}
# membrane properties
_Config['length'] = [35e-6, 35e-6]
_Config['electrode'] = [35e-6, 35e-6]
_Config['nnodes'] = [9, 9]
_Config['thickness'] = [2.2e-6,]
_Config['density'] = [2040,]
_Config['y_modulus'] = [110e9,]
_Config['p_ratio'] = [0.22,]
_Config['isolation'] = 200e-9
_Config['permittivity'] = 6.3
_Config['gap'] = 50e-9
_Config['att_mech'] = 3000
_Config['ndiv'] = [2, 2]
# array properties
_Config['mempitch'] = [45e-6, 45e-6]
_Config['nmem'] = [2, 2]
_Config['nelem'] = 512
_Config['edge_buffer'] = np.sqrt(2 * 40e-6 ** 2)
_Config['taper_radius'] = 3.7125e-3

Config = register_type('Config', _Config)

if __name__ == '__main__':

    import sys
    from cnld import util

    # get script parser and parse arguments
    parser, run_parser = util.script_parser(main, Config)
    args = parser.parse_args()
    array = args.func(args)

    if array is not None:
        if args.file:
            dump(array, args.file)
        else:
            print(array)
            print('Total number of channels ->', sum(get_channel_count(array)))
            print('Number of transmit channels ->', sum(get_channel_count(array, kind='tx')))
            print('Number of receive channels ->', sum(get_channel_count(array, kind='rx')))
            print('Number of transmit/receive channels ->', sum(get_channel_count(array, kind='both')))
