## interaction3 / arrays / foldable_random.py

import numpy as np

from interaction3.abstract import *
from interaction3 import util


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

    dist = util.distance(*np.atleast_2d(pos, all_pos))
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
    random_seed = cfg.random_seed

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
    array0 = _construct_array(0, np.array([-1.25e-3, 0, 0]), _vertices0, elem_pos[mask, :], mem_pos, memprops)

    x0, y0, _ = _vertices1[0]
    x1, y1, _ = _vertices1[2]
    xx, yy, zz = elem_pos.T
    mask = np.logical_and(np.logical_and(np.logical_and(xx >= x0, xx < x1), yy >= y0), yy < y1)
    array1 = _construct_array(1, np.array([0, 0, 0]), _vertices1, elem_pos[mask, :], mem_pos, memprops)

    x0, y0, _ = _vertices2[0]
    x1, y1, _ = _vertices2[2]
    xx, yy, zz = elem_pos.T
    mask = np.logical_and(np.logical_and(np.logical_and(xx >= x0, xx < x1), yy >= y0), yy < y1)
    array2 = _construct_array(2, np.array([1.25e-3, 0, 0]), _vertices2, elem_pos[mask, :], mem_pos, memprops)

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
_Config['edge_buffer'] = 45e-6
_Config['taper_radius'] = 3.75e-3
_Config['random_seed'] = 0

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


