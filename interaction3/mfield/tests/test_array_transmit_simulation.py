
import numpy as np

from interaction3 import bem
from interaction3 import abstract
import interaction3.abstract.arrays.matrix_array
from interaction3.bem.simulations.array_transmit_simulation import ArrayTransmitSimulation
from interaction3.bem.simulations.array_transmit_simulation import connector, get_objects_from_spec

array = abstract.arrays.matrix_array.init(nelem=[2,2])
simulation = abstract.BemArrayTransmitSimulation(frequency=5e6,
                                                 density=1000,
                                                 sound_speed=1500,
                                                 # use_preconditioner=True,
                                                 # use_pressure_load=False,
                                                 # max_level=6,
                                                 # tolerance=0.01,
                                                 # max_iterations=100,
                                                 bounding_box=[-2e-3, -2e-3, 2e-3, 2e-3],
                                                 orders_db='../bem/scripts/orders_dims_0.0040_0.0040.db',
                                                 translations_db='../bem/scripts/translations_dims_0.0040_0.0040.db')

kwargs, meta = connector(simulation, array)

sim = ArrayTransmitSimulation(**kwargs)
sim.solve()