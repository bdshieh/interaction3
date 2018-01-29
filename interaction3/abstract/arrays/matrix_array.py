
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


def init(nmem=(nmem_x, nmem_y), mempitch=(mempitch_x, mempitch_y), nelem=(nelem_x, nelem_y),
         elempitch=(elempitch_x, elempitch_y)):

    nmem_x, nmem_y = nmem
    mempitch_x, mempitch_y = mempitch
    nelem_x, nelem_y = nelem
    elempitch_x, elempitch_y = elempitch


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
            membranes.append(Membrane(id=(i * len(mem_pos) + j),
                                      position=(epos + mpos).tolist()))

        # construct element
        elem = Element(id=i,
                       position=epos.tolist(),
                       membranes=membranes)
        element_position_from_membranes(elem)
        elements.append(elem)

        # construct channel
        chan = Channel(id=i,
                       position=epos.tolist(),
                       elements=elements)
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
        dump(array, open(filename, 'w+'), indent=2)