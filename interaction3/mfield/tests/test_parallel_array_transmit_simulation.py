
import numpy as np

from interaction3 import bem
from interaction3 import abstract
import interaction3.abstract.arrays.matrix_array
from interaction3.bem.simulations.array_transmit_simulation import ArrayTransmitSimulation
from interaction3.bem.simulations.array_transmit_simulation import connector, get_objects_from_spec

array = abstract.arrays.matrix_array.init(nelem=[2,2])
simulation = abstract.BemArrayTransmitSimulation(freqs=[50e3, 5e6, 50e3],
                                                 density=1000,
                                                 sound_speed=1500,
                                                 bounding_box=[-2e-3, -2e-3, 2e-3, 2e-3],
                                                 orders_db='./orders_dims_0.0040_0.0040.db',
                                                 translations_db='./translations_dims_0.0040_0.0040.db')

# kwargs, meta = connector(simulation, array)
# sim = ArrayTransmitSimulation(**kwargs)
# sim.solve()