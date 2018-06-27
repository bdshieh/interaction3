## interaction3 / arrays / foldable_spiral.py

import numpy as np
from scipy.spatial.distance import cdist as distance

from interaction3.abstract import *


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
defaults['att_mech'] = 0
defaults['ndiv'] = [2, 2]

# array properties
defaults['mempitch'] = [45e-6, 45e-6]
defaults['nmem'] = [2, 2]
defaults['nelem'] = 512
defaults['edge_buffer'] = 45e-6
defaults['taper_radius'] = 3.75e-3
defaults['random_seed'] = 0

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


def _check_for_element_collision(pos, all_pos, radius):

    dist = distance(*np.atleast_2d(pos, all_pos))
    if np.any(dist <= 2 * radius):
        return True
    else:
        return False


def _check_for_edge_collision(pos, vertices, edge_buffer):

    x, y, z = pos
    x0, y0, _ = vertices[0]
    x1, y1, _ = vertices[2]

    if (abs(x - x0) >= edge_buffer and abs(x - x1) >= edge_buffer
            and abs(y - y0) >= edge_buffer and abs(y - y1) >= edge_buffer):
        return False
    return True


def init(**kwargs):

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
    random_seed = kwargs['random_seed']

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
    np.random.seed(random_seed)
    collision_radius = np.sqrt((nmem_x * length_x + nmem_x * mempitch_x) ** 2 +
                               (nmem_y * length_y + nmem_y * mempitch_y) ** 2) / 2
    elem_pos = []

    for n in range(nelem):

        while True:

            xx = np.random.rand() * 2 * taper_radius - taper_radius
            yy = np.random.rand() * 2 * taper_radius - taper_radius
            zz = 0
            pos = [xx, yy, zz]

            if not _check_for_edge_collision(pos, _vertices0, edge_buffer):
                if not _check_for_edge_collision(pos, _vertices1, edge_buffer):
                    if not _check_for_edge_collision(pos, _vertices2, edge_buffer):
                        if n == 0: # skip collision check for first element
                            break
                        if not _check_for_element_collision(pos, elem_pos, collision_radius):
                            break

        elem_pos.append([xx, yy, zz])

    elem_pos = np.array(elem_pos)

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

    return array0, array1, array2


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
    parser.add_argument('--ntx', type=int)
    parser.add_argument('--nrx', type=int)
    parser.add_argument('--design-frequency', type=float)
    parser.add_argument('-d', '--dump', nargs='?', default=None)
    parser.set_defaults(**defaults)

    args = vars(parser.parse_args())
    filename = args.pop('dump')

    spec = init(**args)
    print(spec)

    if filename is not None:
        dump(spec, filename, mode='w')
