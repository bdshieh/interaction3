
import numpy as np

from interaction3 import abstract
from interaction3.abstract.arrays import matrix_array
from interaction3.mfield.simulations import TransmitBeamplot


array = abstract.arrays.matrix_array.init(nelem=[2,2])

simulation = abstract.MfieldTransmitBeamplot(sampling_frequency=100e6,
                                             sound_speed=1500,
                                             use_attenuation=True,
                                             frequency_attenuation=0,
                                             attenuation_center_frequency=1e6,
                                             excitation_center_frequecy=5e6,
                                             excitation_bandwidth=4e6,
                                             use_element_factor=False,
                                             element_factor_file='',
                                             field_positions=np.array([[0, 0, 0.05],
                                                                       [0, 0, 0.06],
                                                                       [0, 0, 0.07]])
                                             )

kwargs, meta = TransmitBeamplot.connector(simulation, array)
sim = TransmitBeamplot(**kwargs)
sim.solve()
