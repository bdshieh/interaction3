## interaction3 / arrays / foldable_spiral.py

import numpy as np
from scipy.optimize import brentq

from interaction3.abstract import *
from interaction3 import util


# default parameters
defaults = {}

# membrane properties
defaults['length'] = [35e-6, 35e-6]
defaults['electrode'] = [35e-6, 35e-6]
defaults['nnodes'] = [9, 9]
defaults['thickness'] = [2.2e-6,]
defaults['density'] = [2040,]
defaults['y_modulus'] = [110e9,]
defaults['p_ratio'] = [0.22,]
defaults['isolation'] = 200e-9
defaults['permittivity'] = 6.3
defaults['gap'] = 50e-9
defaults['att_mech'] = 3000
defaults['ndiv'] = [2, 2]

# array properties
defaults['mempitch'] = [45e-6, 45e-6]
defaults['nmem'] = [2, 2]
defaults['nelem'] = 489
defaults['edge_buffer'] = np.sqrt(2 * 40e-6 ** 2)
defaults['taper_radius'] = 3.75e-3 # controls size of spiral
defaults['assert_radius'] = 3.75e-3 - 40e-6

# array pane vertices, hard-coded
_vertices0 = [[-3.75e-3, -3.75e-3, 0],
              [-3.75e-3, 3.75e-3, 0],
              [-1.25e-3, 3.75e-3, 0],
              [-1.25e-3, -3.75e-3, 0]]

_vertices1 = [[-1.25e-3, -3.75e-3, 0],
              [-1.25e-3, 3.75e-3, 0],
              [1.25e-3, 3.75e-3, 0],
              [1.25e-3, -3.75e-3, 0]]

_vertices2 = [[1.25e-3, -3.75e-3, 0],
              [1.25e-3, 3.75e-3, 0],
              [3.75e-3, 3.75e-3, 0],
              [3.75e-3, -3.75e-3, 0]]


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
    # A = 0.42 - 0.5 * cos(2 * pi * (r / (8 * R) + 0.5)) + 0.08 * cos(4 * pi * (r / (8 * R) + 0.5))
    integral = sympy.simplify(sympy.integrate(A * r, (r, 0, r)))
    a_eff = integral.subs(r, R)

    return integral, a_eff


def create(**kwargs):

    # set defaults if not in kwargs:
    for k, v in defaults.items():
        kwargs.setdefault(k, v)

    nelem = kwargs['nelem']
    nmem_x, nmem_y = kwargs['nmem']
    mempitch_x, mempitch_y = kwargs['mempitch']
    length_x, length_y = kwargs['length']
    electrode_x, electrode_y = kwargs['electrode']
    nnodes_x, nnodes_y = kwargs['nnodes']
    ndiv_x, ndiv_y = kwargs['ndiv']
    edge_buffer = kwargs['edge_buffer']
    taper_radius = kwargs['taper_radius']
    assert_radius = kwargs['assert_radius']

    # calculated parameters
    gr = np.pi * (np.sqrt(5) - 1)
    integral, a_eff = blackman_functions(taper_radius)

    # membrane properties
    mem_properties = dict()
    mem_properties['length_x'] = length_x
    mem_properties['length_y'] = length_y
    mem_properties['electrode_x'] = electrode_x
    mem_properties['electrode_y'] = electrode_y
    mem_properties['y_modulus'] = kwargs['y_modulus']
    mem_properties['p_ratio'] = kwargs['p_ratio']
    mem_properties['isolation'] = kwargs['isolation']
    mem_properties['permittivity'] = kwargs['permittivity']
    mem_properties['gap'] = kwargs['gap']
    mem_properties['nnodes_x'] = nnodes_x
    mem_properties['nnodes_y'] = nnodes_y
    mem_properties['thickness'] = kwargs['thickness']
    mem_properties['density'] = kwargs['density']
    mem_properties['att_mech'] = kwargs['att_mech']
    mem_properties['ndiv_x'] = ndiv_x
    mem_properties['ndiv_y'] = ndiv_y

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

    # elem_pos = []
    # n = 0
    # while True:
    #
    #     if len(elem_pos) == nelem:
    #         break
    #
    #     f = lambda r: integral(r) - (2 * n + 1) * a_eff / (2 * nelem)
    #
    #     if n >= nelem:
    #         r = taper_radius
    #     else:
    #         r = brentq(f, 0, taper_radius)
    #     theta = (n + 1) * gr
    #     xx = r * np.sin(theta)
    #     yy = r * np.cos(theta)
    #     zz = 0
    #     n += 1
    #
    #     if _check_for_edge_collision([xx, yy, zz], _vertices0, edge_buffer):
    #         continue
    #     elif _check_for_edge_collision([xx, yy, zz], _vertices1, edge_buffer):
    #         continue
    #     elif _check_for_edge_collision([xx, yy, zz], _vertices2, edge_buffer):
    #         continue
    #     else:
    #         elem_pos.append([xx, yy, zz])
    # elem_pos = np.array(elem_pos)

    # create arrays, bounding box and rotation points are hard-coded
    x0, y0, _ = _vertices0[0]
    x1, y1, _ = _vertices0[2]
    xx, yy, zz = elem_pos.T
    mask = np.logical_and(np.logical_and(np.logical_and(xx >= x0, xx < x1), yy >= y0), yy < y1)
    array0 = _construct_array(0, np.array([-1.25e-3, 0, 0]), _vertices0, elem_pos[mask, :], mem_pos, mem_properties)

    x0, y0, _ = _vertices1[0]
    x1, y1, _ = _vertices1[2]
    xx, yy, zz = elem_pos.T
    mask = np.logical_and(np.logical_and(np.logical_and(xx >= x0, xx < x1), yy >= y0), yy < y1)
    array1 = _construct_array(1, np.array([0, 0, 0]), _vertices1, elem_pos[mask, :], mem_pos, mem_properties)

    x0, y0, _ = _vertices2[0]
    x1, y1, _ = _vertices2[2]
    xx, yy, zz = elem_pos.T
    mask = np.logical_and(np.logical_and(np.logical_and(xx >= x0, xx < x1), yy >= y0), yy < y1)
    array2 = _construct_array(2, np.array([1.25e-3, 0, 0]), _vertices2, elem_pos[mask, :], mem_pos, mem_properties)

    # _assert_radius_rule(assert_radius, array0, array1, array2)

    return array0, array1, array2


def _assert_radius_rule(radius, *arrays):

    pos = np.concatenate(get_channel_positions_from_array(arrays), axis=0)
    r = util.distance(pos, [0,0,0])
    assert np.all(r <= radius)


def _check_for_edge_collision(pos, vertices, edge_buffer):

    x, y, z = pos
    x0, y0, _ = vertices[0]
    x1, y1, _ = vertices[2]

    if (abs(x - x0) >= edge_buffer and abs(x - x1) >= edge_buffer
            and abs(y - y0) >= edge_buffer and abs(y - y1) >= edge_buffer):
        return False
    return True


def _construct_array(id, rotation_origin, vertices, elem_pos, mem_pos, mem_properties):

    if rotation_origin is None:
        rotation_origin = np.array([0,0,0])

    # construct channels
    channels = []
    mem_counter = 0
    elem_counter = 0
    ch_counter = 0

    for e_pos in elem_pos:

        membranes = []
        elements = []

        for m_pos in mem_pos:

            # construct membrane
            m = SquareCmutMembrane(**mem_properties)
            m['id'] = mem_counter
            m['position'] = (e_pos + m_pos).tolist()
            membranes.append(m)
            mem_counter += 1

        # construct element
        elem = Element(id=elem_counter,
                       position=e_pos.tolist(),
                       membranes=membranes)
        element_position_from_membranes(elem)
        elements.append(elem)
        elem_counter += 1

        # construct channel
        ch = Channel(id=ch_counter,
                     kind='both',
                     position=e_pos.tolist(),
                     elements=elements,
                     dc_bias=0,
                     active=True,
                     delay=0)

        channels.append(ch)
        ch_counter += 1

    # construct array
    array = Array(id=id,
                  channels=channels,
                  rotation_origin=rotation_origin.tolist(),
                  vertices=vertices)
    array_position_from_vertices(array)

    return array


## COMMAND LINE INTERFACE ##

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nmem', nargs=2, type=int)
    parser.add_argument('--mempitch', nargs=2, type=float)
    parser.add_argument('--length', nargs=2, type=float)
    parser.add_argument('--electrode', nargs=2, type=float)
    parser.add_argument('--nelem', type=int)
    parser.add_argument('-d', '--dump', nargs='?', default=None)
    parser.set_defaults(**defaults)

    args = vars(parser.parse_args())
    filename = args.pop('dump')

    spec = create(**args)
    print(spec)

    if filename is not None:
        dump(spec, filename, mode='w')

    from matplotlib import pyplot as plt

    pos = np.concatenate(get_membrane_positions_from_array(spec), axis=0)
    plt.plot(pos[:, 0], pos[:, 1], '.')
    plt.gca().set_aspect('equal')
    plt.gca().axvline(-1.25e-3)
    plt.gca().axvline(1.25e-3)
    plt.gca().axvline(-3.75e-3)
    plt.gca().axvline(3.75e-3)
    plt.gca().axhline(-3.75e-3)
    plt.gca().axhline(3.75e-3)
    plt.gca().add_patch(plt.Circle(radius=defaults['assert_radius'], xy=(0,0), fill=None))
    plt.show()