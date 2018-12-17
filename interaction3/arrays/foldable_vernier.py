## interaction3 / arrays / foldable_vernier.py

import numpy as np

from interaction3.abstract import *
from interaction3 import util


def mmain(cfg, args):

    sound_speed = cfg.sound_speed
    design_freq = cfg.design_freq
    ntx = cfg.ntransmit
    nrx = cfg.nreceive
    nmem_x, nmem_y = cfg.nmem
    mempitch_x, mempitch_y = cfg.mempitch
    length_x, length_y = cfg.length
    electrode_x, electrode_y = cfg.electrode
    nnodes_x, nnodes_y = cfg.nnodes
    ndiv_x, ndiv_y = cfg.ndiv
    edge_buffer = cfg.edge_buffer

    # calculated parameters
    p = 3
    d = sound_speed / design_freq / 2 * 0.9
    tx_pitch = p * d
    rx_pitch = (p - 1) * d
    tx_r = ntx / 2 * tx_pitch  # + 0.00025
    rx_r = nrx / 2 * rx_pitch  # + 0.00017

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
    xx, yy, zz = np.meshgrid(np.linspace(0, (ntx - 1) * tx_pitch, ntx),
                             np.linspace(0, (ntx - 1) * tx_pitch, ntx),
                             0)
    tx_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] - [(ntx - 1) * tx_pitch / 2,
                                                          (ntx - 1) * tx_pitch / 2,
                                                          0]

    # define receive element centers
    xx, yy, zz = np.meshgrid(np.linspace(0, (nrx - 1) * rx_pitch, nrx),
                             np.linspace(0, (nrx - 1) * rx_pitch, nrx),
                             0)
    rx_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] - [(nrx - 1) * rx_pitch / 2,
                                                          (nrx - 1) * rx_pitch / 2,
                                                          0]

    # taper transmit corner elements
    dist = util.distance(tx_pos, np.array([[0, 0, 0]]))
    mask = dist <= tx_r
    tx_pos = tx_pos[mask.squeeze(), :]

    # taper receive corner elements
    dist = util.distance(rx_pos, np.array([[0, 0, 0]]))
    mask = dist <= rx_r
    rx_pos = rx_pos[mask.squeeze(), :]

    # create arrays, bounding box and rotation points are hard-coded
    vertices = [[-3.75e-3, -3.75e-3, 0],
                [-3.75e-3, 3.75e-3, 0],
                [-1.25e-3, 3.75e-3, 0],
                [-1.25e-3, -3.75e-3, 0]]
    x0, y0, _ = vertices[0]
    x1, y1, _ = vertices[2]
    xx, yy, zz = tx_pos.T
    tx_mask = np.logical_and(np.logical_and(np.logical_and(xx >= (x0 + edge_buffer), xx < (x1 - edge_buffer)),
                                         yy >= (y0 + edge_buffer)), yy < (y1 - edge_buffer))
    xx, yy, zz = rx_pos.T
    rx_mask = np.logical_and(np.logical_and(np.logical_and(xx >= (x0 + edge_buffer), xx < (x1 - edge_buffer)),
                                         yy >= (y0 + edge_buffer)), yy < (y1 - edge_buffer))
    array0 = _construct_array(0, np.array([-1.25e-3, 0, 0]), vertices, tx_pos[tx_mask, :], rx_pos[rx_mask, :], mem_pos,
                              mem_properties)

    vertices = [[-1.25e-3, -3.75e-3, 0],
                [-1.25e-3, 3.75e-3, 0],
                [1.25e-3, 3.75e-3, 0],
                [1.25e-3, -3.75e-3, 0]]
    x0, y0, _ = vertices[0]
    x1, y1, _ = vertices[2]
    xx, yy, zz = tx_pos.T
    tx_mask = np.logical_and(np.logical_and(np.logical_and(xx >= (x0 + edge_buffer), xx < (x1 - edge_buffer)),
                                         yy >= (y0 + edge_buffer)), yy < (y1 - edge_buffer))
    xx, yy, zz = rx_pos.T
    rx_mask = np.logical_and(np.logical_and(np.logical_and(xx >= (x0 + edge_buffer), xx < (x1 - edge_buffer)),
                                         yy >= (y0 + edge_buffer)), yy < (y1 - edge_buffer))
    array1 = _construct_array(1, np.array([0, 0, 0]), vertices, tx_pos[tx_mask, :], rx_pos[rx_mask, :], mem_pos,
                              mem_properties)

    vertices = [[1.25e-3, -3.75e-3, 0],
                [1.25e-3, 3.75e-3, 0],
                [3.75e-3, 3.75e-3, 0],
                [3.75e-3, -3.75e-3, 0]]
    x0, y0, _ = vertices[0]
    x1, y1, _ = vertices[2]
    xx, yy, zz = tx_pos.T
    tx_mask = np.logical_and(np.logical_and(np.logical_and(xx >= (x0 + edge_buffer), xx < (x1 - edge_buffer)),
                                         yy >= (y0 + edge_buffer)), yy < (y1 - edge_buffer))
    xx, yy, zz = rx_pos.T
    rx_mask = np.logical_and(np.logical_and(np.logical_and(xx >= (x0 + edge_buffer), xx < (x1 - edge_buffer)),
                                         yy >= (y0 + edge_buffer)), yy < (y1 - edge_buffer))
    array2 = _construct_array(2, np.array([1.25e-3, 0, 0]), vertices, tx_pos[tx_mask, :], rx_pos[rx_mask, :], mem_pos,
                              mem_properties)

    return array0, array1, array2


def _construct_array(id, rotation_origin, vertices, tx_pos, rx_pos, mem_pos, mem_properties):

    if rotation_origin is None:
        rotation_origin = np.array([0,0,0])

    # construct channels
    elements = []
    mem_counter = 0
    elem_counter = 0

    for e_pos in tx_pos:
        membranes = []
        for m_pos in mem_pos:
            # construct membrane
            m = SquareCmutMembrane(**memprops)
            m.id = mem_counter
            m.position = (epos + mpos).tolist()

            membranes.append(m)
            mem_counter += 1

        # construct element
        if np.any(np.all(np.isclose(e_pos, rx_pos), axis=1)):
            kind = 'both'
        else:
            kind = 'transmit'

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

    for e_pos in rx_pos:

        if np.any(np.all(np.isclose(e_pos, tx_pos), axis=1)):
            continue

        membranes = []
        for m_pos in mem_pos:
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
_Config['ntransmit'] = 25
_Config['nreceive'] = 25
_Config['design_freq'] = 7e6
_Config['sound_speed'] = 1540
_Config['edge_buffer'] = 0  # np.sqrt(2 * 40e-6 ** 2)

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
