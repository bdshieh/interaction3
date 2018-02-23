## interaction3 / abstract/ arrays/ matrix_array.py

import numpy as np

from interaction3.abstract import *

# default parameters
defaults = dict()

# membrane properties
defaults['mempitch'] = [50e-6, 50e-6]
defaults['nmem'] = [2, 2]
defaults['length'] = [40e-6, 40e-6]
defaults['electrode'] = [40e-6, 40e-6]
defaults['nnodes'] = [9, 9]
defaults['thickness'] = [2.2e-6,]
defaults['density'] = [2040,]
defaults['y_modulus'] = [110e9,]
defaults['p_ratio'] = [0.22,]
defaults['isolation'] = 200e-9
defaults['permittivity'] = 6.3
defaults['gap'] = 100e-9
defaults['att_mech'] = 0
defaults['ndiv'] = [2, 2]

# array properties
defaults['elempitch'] = [100e-6, 100e-6]
defaults['nelem'] = [7, 7]


def init(**kwargs):

    # set defaults if not in kwargs:
    for k, v in defaults.items():
        kwargs.setdefault(k, v)

    nmem_x, nmem_y = kwargs['nmem']
    mempitch_x, mempitch_y = kwargs['mempitch']
    length_x, length_y = kwargs['length']
    electrode_x, electrode_y = kwargs['electrode']
    nnodes_x, nnodes_y = kwargs['nnodes']
    ndiv_x, ndiv_y = kwargs['ndiv']
    nelem_x, nelem_y = kwargs['nelem']
    elempitch_x, elempitch_y = kwargs['elempitch']

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

    # calculate element positions
    xx, yy, zz = np.meshgrid(np.linspace(0, (nelem_x - 1) * elempitch_x, nelem_x),
                             np.linspace(0, (nelem_y - 1) * elempitch_y, nelem_y),
                             0)
    elem_pos = np.c_[xx.ravel(), yy.ravel(), zz.ravel()] - [(nelem_x - 1) * elempitch_x / 2,
                                                           (nelem_y - 1) * elempitch_y / 2,
                                                           0]

    # construct channels
    channels = []

    for i, epos in enumerate(elem_pos):

        membranes = []
        elements = []

        for j, mpos in enumerate(mem_pos):

            # construct membrane
            m = SquareCmutMembrane(**mem_properties)
            m['id'] = i * len(mem_pos) + j
            m['position'] = position=(epos + mpos).tolist()
            membranes.append(m)
            # membranes.append(SquareCmutMembrane(id=(i * len(mem_pos) + j),
            #                                     position=(epos + mpos).tolist(),
            #                                     **mem_properties))

        # construct element
        elem = Element(id=i,
                       position=epos.tolist(),
                       membranes=membranes)
        element_position_from_membranes(elem)
        elements.append(elem)

        # construct channel
        chan = Channel(id=i,
                       kind='both',
                       position=epos.tolist(),
                       elements=elements,
                       dc_bias=0,
                       active=True,
                       delay=0)

        # channel_position_from_elements(chan)
        channels.append(chan)

    # construct array
    array = Array(id=0,
                  channels=channels,
                  position=[0, 0, 0])

    return array


## COMMAND LINE INTERFACE ##

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-nmem', '--nmem', nargs=2, type=int)
    parser.add_argument('-mempitch', '--mempitch', nargs=2, type=float)
    parser.add_argument('-nelem', '--nelem', nargs=2, type=int)
    parser.add_argument('-elempitch', '--elempitch', nargs=2, type=float)
    parser.add_argument('-d', '--dump-json', nargs='?', default=None)
    parser.set_defaults(**defaults)

    args = vars(parser.parse_args())
    filename = args.pop('dump_json')

    spec = init(**args)
    print(spec)

    if filename is not None:
        dump(spec, filename)