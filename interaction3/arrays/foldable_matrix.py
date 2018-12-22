## interaction3 / arrays / foldable_matrix.py

import numpy as np

from interaction3.abstract import *
from interaction3 import util


def main(cfg, args):

    nmem_x, nmem_y = cfg.nmem
    mempitch_x, mempitch_y = cfg.mempitch
    length_x, length_y = cfg.length
    electrode_x, electrode_y = cfg.electrode
    npatch_x, npatch_y = cfg.npatch
    nnodes_x, nnodes_y = cfg.nnodes
    ndiv_x, ndiv_y = cfg.ndiv
    nelem_x, nelem_y = cfg.nelem
    elempitch_x, elempitch_y = cfg.elempitch
    edge_buffer = cfg.edge_buffer
    taper_radius = cfg.taper_radius

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

    # calculate element positions
    xx, yy, zz = np.meshgrid(np.linspace(0, (nelem_x - 1) * elempitch_x, nelem_x),
                             np.linspace(0, (nelem_y - 1) * elempitch_y, nelem_y),
                             0)
    elem_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] - [(nelem_x - 1) * elempitch_x / 2,
                                                           (nelem_y - 1) * elempitch_y / 2,
                                                           0]

    # taper transmit corner elements
    dist = util.distance(elem_pos, np.array([[0, 0, 0]]))
    mask = (dist <= taper_radius).squeeze()
    elem_pos = elem_pos[mask, :]

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
        # construct membrane list
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
_Config['elempitch'] = [90e-6, 90e-6]
_Config['nelem'] = [80, 80]
_Config['edge_buffer'] = 40e-6  # np.sqrt(2 * 40e-6 ** 2)
_Config['taper_radius'] = 3.7125e-3

Config = register_type('Config', _Config)

if __name__ == '__main__':

    import sys
    from interaction3 import util

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

