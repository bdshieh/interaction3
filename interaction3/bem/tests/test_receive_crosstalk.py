## interaction3 / bem / tests / test_receive_crosstalk.py

from interaction3 import abstract
import interaction3.abstract.arrays.matrix_array
from interaction3.bem.solvers import ReceiveCrosstalk

array = abstract.arrays.matrix_array.init(nelem=[4, 4])
simulation = abstract.BemSimulation(frequency=5e6,
                                    plane_wave_vector=[0, 0, -1],
                                    plane_wave_pressure=1,
                                    density=1000,
                                    sound_speed=1500,
                                    max_level=4,
                                    bounding_box=[-1e-3, -1e-3, 1e-3, 1e-3],
                                    orders_db='./orders_dims_0.0020_0.0020.db',
                                    translations_db='./translations_dims_0.0020_0.0020.db')

kwargs, meta = ReceiveCrosstalk.connector(simulation, array)
solver = ReceiveCrosstalk(**kwargs)
solver.solve()
