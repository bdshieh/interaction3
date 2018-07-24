## interaction3 / bem / tests / test_transmit_crosstalk.py

from interaction3 import abstract
import interaction3.abstract.arrays.matrix_array
from interaction3.bem.solvers import TransmitCrosstalk

array = abstract.arrays.matrix_array.init(nelem=[2,2])
simulation = abstract.BemSimulation(frequency=1e6,
                                    density=1000,
                                    sound_speed=1500,
                                    max_level=5,
                                    tolerance=0.001,
                                    bounding_box=[-1e-3, -1e-3, 1e-3, 1e-3],
                                    orders_db='./orders_dims_0.0020_0.0020.db',
                                    translations_db='./translations_dims_0.0020_0.0020.db')

kwargs, meta =TransmitCrosstalk.connector(simulation, array)
solver = TransmitCrosstalk(**kwargs)
solver.solve()
