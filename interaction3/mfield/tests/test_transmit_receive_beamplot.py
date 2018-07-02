
import numpy as np

from interaction3 import abstract
from interaction3.arrays import matrix
from interaction3.mfield.solvers.transmit_receive_beamplot_2 import TransmitReceiveBeamplot2


array = matrix.create(nelem=[2, 2])

simulation = abstract.MfieldSimulation(sampling_frequency=100e6,
                                       sound_speed=1540,
                                       excitation_center_frequecy=5e6,
                                       excitation_bandwidth=4e6,
                                       field_positions=np.array([[0, 0, 0.05],
                                                                 [0, 0, 0.06],
                                                                 [0, 0, 0.07]])
                                       )

kwargs, meta = TransmitReceiveBeamplot2.connector(simulation, array)
sim = TransmitReceiveBeamplot2(**kwargs)
sim.solve()

rf_data = sim.result['rf_data']
times = sim.result['times']

