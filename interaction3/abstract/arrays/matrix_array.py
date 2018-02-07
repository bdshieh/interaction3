
from interaction3.abstract.classes import *
from interaction3.abstract.manipulations import *


# default parameters
mempitch_x = 52e-6
mempitch_y = 52e-6
nmem_x = 2
nmem_y = 2
elempitch_x = 104e-6
elempitch_y = 104e-6
nelem_x = 7
nelem_y = 7
length_x = 40e-6
length_y = 40e-6
electrode_x = 40e-6
electrode_y = 40e-6
nnodes_x = 9
nnodes_y = 9
thickness = [2.2e-6,]
density = [2040,]
y_modulus = [110e9,]
p_ratio = [0.22,]
isolation = 220e-9
permittivity = 6.3
gap = 100e-9
att_mech = 0



def init(nmem=(nmem_x, nmem_y),
         mempitch=(mempitch_x, mempitch_y),
         nelem=(nelem_x, nelem_y),
         elempitch=(elempitch_x, elempitch_y),
         length=(length_x, length_y),
         electrode=(electrode_x, electrode_y),
         nnodes=(nnodes_x, nnodes_y),
         y_modulus=y_modulus,
         p_ratio=p_ratio,
         isolation=isolation,
         permittivity=permittivity,
         gap=gap,
         thickness=thickness,
         density=density,
         att_mech=att_mech
         ):

    nmem_x, nmem_y = nmem
    mempitch_x, mempitch_y = mempitch
    nelem_x, nelem_y = nelem
    elempitch_x, elempitch_y = elempitch
    length_x, length_y = length
    electrode_x, electrode_y = electrode
    nnodes_x, nnodes_y = nnodes

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
            membranes.append(SquareCmutMembrane(id=(i * len(mem_pos) + j),
                                                position=(epos + mpos).tolist(),
                                                length_x=length_x,
                                                length_y=length_y,
                                                electrode_x=electrode_x,
                                                electrode_y=electrode_y,
                                                y_modulus=y_modulus,
                                                p_ratio=p_ratio,
                                                isolation=isolation,
                                                permittivity=permittivity,
                                                gap=gap,
                                                nnodes_x=nnodes_x,
                                                nnodes_y=nnodes_y,
                                                thickness=thickness,
                                                density=density,
                                                att_mech=att_mech))

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
                       dc_bias=5,
                       delay=0)

        # channel_position_from_elements(chan)
        channels.append(chan)

    # construct array
    array = Array(id=0,
                  channels=channels)

    return array


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-nmem', '--nmem', nargs=2, type=int, default=[nmem_x, nmem_y])
    parser.add_argument('-mempitch', '--mempitch', nargs=2, type=float, default=[mempitch_x, mempitch_y])
    parser.add_argument('-nelem', '--nelem', nargs=2, type=int, default=[nelem_x, nelem_y])
    parser.add_argument('-elempitch', '--elempitch', nargs=2, type=float, default=[elempitch_x, elempitch_y])
    parser.add_argument('-d', '--dump-json', nargs='?', default=None)

    args = vars(parser.parse_args())
    filename = args.pop('dump_json')

    array = init(**args)
    print(array)

    if filename is not None:
        dump(array, filename)