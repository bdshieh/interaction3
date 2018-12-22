''' 
Abstract representation of a matrix array.
'''
import numpy as np

from interaction3.abstract import *
from interaction3 import util


def main(cfg, args):

    nmem_x, nmem_y = cfg.nmem
    mempitch_x, mempitch_y = cfg.mempitch
    length_x, length_y = cfg.length
    electrode_x, electrode_y = cfg.electrode
    nelem_x, nelem_y = cfg.nelem
    elempitch_x, elempitch_y = cfg.elempitch
    ndiv_x, ndiv_y = cfg.ndiv

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
    memprops['ndiv_x'] = ndiv_x
    memprops['ndiv_y'] = ndiv_y
    memprops['thickness'] = cfg.thickness
    memprops['density'] = cfg.density
    memprops['att_mech'] = cfg.att_mech
    memprops['kmat_file'] = cfg.kmat_file

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

    # construct element list
    elements = []
    elem_counter = 0
    mem_counter = 0

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
        elem.position = epos.tolist()
        elem.kind = 'both'
        elem.membranes = membranes
        # element_position_from_membranes(elem)
        elements.append(elem)
        elem_counter += 1

    # construct array
    array = Array()
    array.id = 0
    array.elements = elements
    array.position = [0, 0, 0]

    return array


# default configuration
_Config = {}
# membrane properties
_Config['length'] = [40e-6, 40e-6]
_Config['electrode'] = [40e-6, 40e-6]
_Config['thickness'] = [2e-6,]
_Config['density'] = [2040,]
_Config['y_modulus'] = [110e9,]
_Config['p_ratio'] = [0.22,]
_Config['isolation'] = 200e-9
_Config['permittivity'] = 6.3
_Config['gap'] = 100e-9
_Config['att_mech'] = 0
_Config['ndiv'] = [2, 2]
_Config['kmat_file'] = ''
# array properties
_Config['mempitch'] = [60e-6, 60e-6]
_Config['nmem'] = [1, 1]
_Config['elempitch'] = [60e-6, 60e-6]
_Config['nelem'] = [5, 5]

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
    